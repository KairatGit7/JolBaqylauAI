import math
from pathlib import Path
from xml.etree.ElementTree import ParseError

from flask import Blueprint, abort, current_app, flash, jsonify, redirect, render_template, request, url_for
from sqlalchemy.exc import IntegrityError

from .gps import coordinates, parse_track, synchronize
from .models import db, Video, DetectionEvent, RoadSegment, GPSObservation, active_events
from .routes import login_required
from .vision import LABELS

bp = Blueprint('mapping', __name__)


@bp.get('/map')
@login_required
def map_page():
    return render_template('map.html', segments=RoadSegment.query.order_by(RoadSegment.name).all(),
                           unlocated=DetectionEvent.query.filter(active_events(), DetectionEvent.latitude.is_(None)).count())


@bp.get('/api/events')
@login_required
def events():
    query = DetectionEvent.query.filter(active_events())
    if request.args.get('category'):
        if request.args['category'] not in LABELS:
            abort(400)
        query = query.filter_by(category=request.args['category'])
    if request.args.get('status'):
        if request.args['status'] not in {'predicted', 'confirmed', 'rejected'}:
            abort(400)
        query = query.filter_by(review_status=request.args['status'])
    rows = query.order_by(DetectionEvent.id.desc()).all()
    return jsonify([dict(id=e.id, category=e.category, label=LABELS[e.category], confidence=e.confidence,
                         width=e.video.width, height=e.video.height, filter_status=e.filter_status,
                         latitude=e.latitude, longitude=e.longitude, location_source=e.location_source,
                         segment=e.segment.name if e.segment else None, review_status=e.review_status, is_demo=e.video.is_demo,
                         seconds=e.start_seconds, observed_at=e.video.observed_at.isoformat() + 'Z',
                         evidence=url_for('web.evidence', filename=e.evidence_name),
                         detail=url_for('web.video_detail', video_id=e.video_id) + f'#event-{e.id}') for e in rows])


@bp.post('/segments')
@login_required
def segment_create():
    name = request.form.get('name', '').strip()
    description = request.form.get('description', '').strip()
    if not name or len(name) > 200 or len(description) > 2000:
        abort(400)
    db.session.add(RoadSegment(name=name, description=description))
    try:
        db.session.commit()
        flash('Жол учаскесі сақталды. Координат автоматты берілмейді.', 'success')
    except IntegrityError:
        db.session.rollback()
        flash('Мұндай атаулы учаске бар.', 'error')
    return redirect(url_for('mapping.map_page'))


@bp.post('/videos/<video_id>/gps')
@login_required
def import_gps(video_id):
    video = db.get_or_404(Video, video_id)
    if video.job.status != 'completed':
        flash('GPS импортын талдау аяқталғаннан кейін орындаңыз.', 'error')
        return redirect(url_for('web.video_detail', video_id=video.id))
    upload = request.files.get('gps')
    try:
        if not upload or request.form.get('verified') != 'yes':
            raise ValueError('GPS файлын және оның осы сапарға тиесілі екенін растаңыз.')
        offset = float(request.form.get('offset', '0'))
        if not math.isfinite(offset) or abs(offset) > 86400:
            raise ValueError('Уақыт ығысуы жарамсыз; ең көбі ±86400 с.')
        source = request.form.get('source', '').strip()
        if not source or len(source) > 200:
            raise ValueError('GPS дереккөзін көрсетіңіз (200 таңбаға дейін).')
        points = parse_track(upload.stream.read(2 * 1024 * 1024 + 1),
                             Path(upload.filename or '').suffix.lower(), video.observed_at)
        GPSObservation.query.filter_by(video_id=video.id).delete()
        for seconds, lat, lon in points:
            db.session.add(GPSObservation(video_id=video.id, track_seconds=seconds, latitude=lat, longitude=lon))
        video.gps_offset, video.gps_source = offset, source
        located = 0
        for e in video.events:
            if e.location_source and e.location_source.startswith('Қолмен:'):
                continue
            # Ең сенімді дәлел кадрының көлік координатасы.
            timestamp = max(e.hits, key=lambda hit: hit['confidence'])['timestamp']
            location = synchronize(points, timestamp, offset)
            e.latitude, e.longitude = location if location else (None, None)
            e.location_source = f'GPS: {source}; ығысу {offset:+g} с' if location else None
            located += location is not None
        db.session.commit()
        flash(f'GPS сақталды. Орны сәйкестендірілген оқиғалар: {located}.', 'success')
    except (ValueError, UnicodeError, TypeError, KeyError, ParseError) as exc:
        db.session.rollback()
        flash(f'GPS импорты орындалмады: {exc}', 'error')
        return render_template('video.html', video=video, segments=RoadSegment.query.all()), 400
    return redirect(url_for('web.video_detail', video_id=video.id))


@bp.post('/events/<int:event_id>/location')
@login_required
def assign_location(event_id):
    e = db.get_or_404(DetectionEvent, event_id)
    try:
        segment_id = request.form.get('segment_id')
        e.segment = db.get_or_404(RoadSegment, int(segment_id)) if segment_id else None
        lat, lon = request.form.get('latitude', '').strip(), request.form.get('longitude', '').strip()
        if lat or lon:
            source = request.form.get('source', '').strip()
            if request.form.get('verified') != 'yes' or not source or len(source) > 250:
                raise ValueError('Координат дереккөзін және тексерілгенін растаңыз.')
            e.latitude, e.longitude = coordinates(lat, lon)
            e.location_source = 'Қолмен: ' + source
        if request.form.get('clear_location') == 'yes':
            e.latitude = e.longitude = e.location_source = None
        db.session.commit()
        flash('Учаске мен орын деректері сақталды.', 'success')
    except (ValueError, TypeError) as exc:
        db.session.rollback()
        flash(str(exc), 'error')
        return render_template('video.html', video=e.video, segments=RoadSegment.query.all()), 400
    return redirect(url_for('web.video_detail', video_id=e.video_id) + f'#event-{e.id}')
