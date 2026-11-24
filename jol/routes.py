import hmac
import hashlib
import secrets
import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import Blueprint, abort, current_app, flash, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.security import check_password_hash
from sqlalchemy.exc import SQLAlchemyError

from .models import db, Video, ProcessingJob, DetectionEvent, RoadSegment, now, active_events
from .vision import inspect_video, LABELS
from .presets import inference_settings

bp = Blueprint('web', __name__)


def csrf_token():
    if 'csrf' not in session:
        session['csrf'] = secrets.token_urlsafe(32)
    return session['csrf']


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get('admin'):
            if request.path.startswith('/api/'):
                return jsonify(error='Жүйеге кіріңіз.'), 401
            return redirect(url_for('web.login'))
        return view(*args, **kwargs)
    return wrapped


def visitor_key():
    if 'visitor_key' not in session:
        session['visitor_key'] = secrets.token_hex(32)
    return session['visitor_key']


def visible_videos():
    query = Video.query
    return query if session.get('admin') else query.filter_by(owner_key=visitor_key())


def can_access(record):
    return bool(session.get('admin') or (record and record.owner_key
                and record.owner_key == session.get('visitor_key')))


def video_access_required(view):
    @wraps(view)
    def wrapped(video_id, *args, **kwargs):
        video = db.session.get(Video, video_id)
        if not can_access(video):
            if request.path.startswith('/api/'):
                return jsonify(error='Бұл жазбаға қол жеткізуге рұқсат жоқ.'), 401
            return redirect(url_for('web.login'))
        return view(video_id, *args, **kwargs)
    return wrapped


@bp.before_app_request
def csrf_check():
    if request.method == 'POST':
        token = request.form.get('csrf_token') or request.headers.get('X-CSRF-Token', '')
        if not session.get('csrf') or not token or not hmac.compare_digest(str(session['csrf']), token):
            abort(400)


@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        attempts = current_app.extensions.setdefault('login_attempts', {})
        key = request.remote_addr
        recent = [t for t in attempts.get(key, []) if time.monotonic() - t < 300]
        attempts[key] = recent
        if len(recent) >= 10:
            flash('Кіру әрекеті тым көп. 5 минуттан кейін қайталаңыз.', 'error')
            return render_template('login.html'), 429
        if check_password_hash(current_app.config['ADMIN_PASSWORD_HASH'], request.form.get('password', '')):
            owner = session.get('visitor_key')
            session.clear()
            session['admin'] = True
            if owner:
                session['visitor_key'] = owner
            attempts.pop(key, None)
            return redirect(url_for('web.home'))
        recent.append(time.monotonic())
        flash('Құпиясөз дұрыс емес.', 'error')
    return render_template('login.html')


@bp.post('/logout')
def logout():
    owner = session.get('visitor_key')
    session.clear()
    if owner:
        session['visitor_key'] = owner
    return redirect(url_for('web.home'))


@bp.get('/')
def home():
    if not session.get('admin'):
        videos = visible_videos().order_by(Video.created_at.desc()).limit(6).all()
        events = DetectionEvent.query.join(Video).filter(Video.owner_key == visitor_key(), active_events()).all()
        return render_template('home.html', total=len(events), located=sum(e.latitude is not None for e in events),
                               category_counts={key: sum(e.category == key for e in events) for key in LABELS},
                               demo_events=sum(e.video.is_demo for e in events), segment_count=0,
                               confirmed=sum(e.review_status == 'confirmed' for e in events), videos=videos,
                               active=visible_videos().join(ProcessingJob).filter(ProcessingJob.status.in_(['queued', 'processing'])).count())
    return render_template('home.html', total=DetectionEvent.query.filter(active_events()).count(),
                           located=DetectionEvent.query.filter(active_events(), DetectionEvent.latitude.is_not(None), DetectionEvent.longitude.is_not(None)).count(),
                           category_counts=dict(db.session.query(DetectionEvent.category, db.func.count(DetectionEvent.id)).filter(active_events()).group_by(DetectionEvent.category).all()),
                           demo_events=DetectionEvent.query.join(Video).filter(active_events(), Video.is_demo.is_(True)).count(),
                           segment_count=db.session.query(DetectionEvent.segment_id).filter(active_events(), DetectionEvent.segment_id.is_not(None)).distinct().count(),
                           confirmed=DetectionEvent.query.filter(active_events()).filter_by(review_status='confirmed').count(),
                           videos=Video.query.order_by(Video.created_at.desc()).limit(6).all(),
                           active=ProcessingJob.query.filter(ProcessingJob.status.in_(['queued', 'processing'])).count())


def analysis_settings(fps):
    return dict(inference_settings(fps), model_revision=current_app.config['MODEL_REVISION'],
                pipeline_version='temporal-v1')


def matching_analysis(digest, settings):
    return visible_videos().filter_by(sha256=digest, **settings).join(ProcessingJob).filter(
        ProcessingJob.status != 'failed').order_by(Video.created_at.desc()).first()


@bp.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if request.method == 'POST':
        if ProcessingJob.query.filter(ProcessingJob.status.in_(['queued', 'processing'])).count() >= 4:
            flash('Кезек толы. Ағымдағы талдау аяқталғанша күтіңіз.', 'error')
            return redirect(url_for('web.analysis'))
        upload = request.files.get('video')
        extension = Path(upload.filename or '').suffix.lower() if upload else ''
        if not upload or extension not in {'.mp4', '.mov', '.avi'}:
            flash('MP4, MOV немесе AVI бейнесін таңдаңыз.', 'error')
            return render_template('analysis.html', videos=visible_videos().all()), 400
        video_id = uuid.uuid4().hex
        filename = video_id + extension
        path = current_app.config['STORAGE'] / 'raw' / filename
        try:
            observed_at = datetime.fromisoformat(request.form.get('observed_at', ''))
            # Интерфейс Талдықорғанның UTC+5 уақытын қабылдайды; БД UTC сақтайды.
            from datetime import timedelta
            observed_at = observed_at.replace(tzinfo=timezone(timedelta(hours=5))).astimezone(timezone.utc).replace(tzinfo=None)
            if observed_at > now():
                raise ValueError('Түсірілім күні болашақта болмауы тиіс.')
            upload.save(path)
            path.chmod(0o600)
            with path.open('rb') as source:
                digest = hashlib.file_digest(source, 'sha256').hexdigest()
            metadata = inspect_video(path, current_app.config['MAX_VIDEO_SECONDS'])
            settings = analysis_settings(metadata['fps'])
            duplicate = matching_analysis(digest, settings)
            if duplicate:
                path.unlink(missing_ok=True)
                flash('Бұл бейне бұрын жүктелген. Бар жазба ашылды.', 'success')
                return redirect(url_for('web.video_detail', video_id=duplicate.id))
            video = Video(id=video_id, name=(upload.filename or 'Бейне')[:200], stored_name=filename,
                           owner_key=visitor_key(),
                           sha256=digest, is_current=False, is_demo=request.form.get('is_demo') == 'yes',
                           observed_at=observed_at, **settings, **metadata)
            db.session.add(video)
            db.session.add(ProcessingJob(video=video))
            db.session.commit()
            if current_app.config['START_WORKER']:
                current_app.extensions['job_runner'].submit(video.id)
            return redirect(url_for('web.video_detail', video_id=video.id))
        except (ValueError, OSError) as exc:
            db.session.rollback()
            path.unlink(missing_ok=True)
            flash(str(exc), 'error')
            return render_template('analysis.html', videos=visible_videos().all()), 400
    return render_template('analysis.html', videos=visible_videos().order_by(Video.created_at.desc()).all())


@bp.post('/videos/<video_id>/reanalyze')
@video_access_required
def reanalyze(video_id):
    source = db.get_or_404(Video, video_id)
    if ProcessingJob.query.filter(ProcessingJob.status.in_(['queued', 'processing'])).count() >= 4:
        flash('Кезек толы. Ағымдағы талдау аяқталғанша күтіңіз.', 'error')
        return redirect(url_for('web.video_detail', video_id=source.id))
    try:
        settings = analysis_settings(source.fps)
        with (current_app.config['STORAGE'] / 'raw' / source.stored_name).open('rb') as raw:
            digest = hashlib.file_digest(raw, 'sha256').hexdigest()
        duplicate = matching_analysis(digest, settings)
        if duplicate:
            flash('Осы файл мен параметрлердің бар талдауы ашылды.', 'success')
            return redirect(url_for('web.video_detail', video_id=duplicate.id))
        video = Video(id=uuid.uuid4().hex, name=source.name, stored_name=source.stored_name,
                      owner_key=source.owner_key,
                      sha256=digest, is_current=False, is_demo=source.is_demo, observed_at=source.observed_at,
                      duration=source.duration, fps=source.fps, width=source.width, height=source.height, **settings)
        db.session.add(video)
        db.session.add(ProcessingJob(video=video))
        db.session.commit()
        if current_app.config['START_WORKER']:
            current_app.extensions['job_runner'].submit(video.id)
        flash('Жаңа талдау басталды. Бұрынғы нәтиже мен адам тексеруі сақталды.', 'success')
        return redirect(url_for('web.video_detail', video_id=video.id))
    except (ValueError, OSError) as exc:
        db.session.rollback()
        flash(f'Қайта талдау басталмады: {exc}', 'error')
        return redirect(url_for('web.video_detail', video_id=source.id))


@bp.get('/videos/<video_id>')
@video_access_required
def video_detail(video_id):
    return render_template('video.html', video=db.get_or_404(Video, video_id), segments=RoadSegment.query.order_by(RoadSegment.name).all())


@bp.get('/api/videos/<video_id>/status')
@video_access_required
def video_status(video_id):
    job = db.get_or_404(Video, video_id).job
    return jsonify(status=job.status, progress=job.progress, frames=job.frames_analyzed, error=job.error)


@bp.get('/evidence/<path:filename>')
def evidence(filename):
    if not session.get('admin'):
        video = db.session.get(Video, filename.split('/')[0])
        if not can_access(video) or '..' in filename.split('/'):
            return redirect(url_for('web.login'))
    response = send_from_directory(current_app.config['STORAGE'] / 'evidence', filename, conditional=True)
    response.headers['Cache-Control'] = 'private, no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


@bp.get('/raw/<video_id>')
@video_access_required
def raw_video(video_id):
    video = db.get_or_404(Video, video_id)
    response = send_from_directory(current_app.config['STORAGE'] / 'raw', video.stored_name, as_attachment=True)
    response.headers['Cache-Control'] = 'private, no-store'
    return response


@bp.post('/events/<int:event_id>/review')
def review(event_id):
    item = db.session.get(DetectionEvent, event_id)
    if not can_access(item.video if item else None):
        return redirect(url_for('web.login'))
    event = db.get_or_404(DetectionEvent, event_id)
    saved = False
    status = request.form.get('verdict') or request.form.get('status')
    note = request.form.get('note', '').strip()
    if status not in {'predicted', 'confirmed', 'rejected'} or len(note) > 2000:
        abort(400)
    if status != 'predicted' and not note:
        flash('Тексеру негізін жазыңыз.', 'error')
    else:
        from .reviews import archive_review
        image_path = None
        try:
            example, image_path = archive_review(event, status, note, request.form.get('reason', 'other'))
            db.session.add(example)
            event.review_status = status
            event.review_note = note
            event.reviewed_at = now()
            db.session.commit()
            saved = True
            flash('Тексеру нәтижесі және бөлек бағалау үлгісі сақталды.', 'success')
        except (ValueError, OSError, SQLAlchemyError) as exc:
            db.session.rollback()
            if image_path:
                image_path.unlink(missing_ok=True)
            flash(f'Тексеру сақталмады: {exc}', 'error')
    return redirect(url_for('web.video_detail', video_id=event.video_id) + (f'#event-{event.id}' if saved else ''))


@bp.get('/about')
def about():
    return render_template('about.html')
