"""Small, private, fully labelled frame evaluation; never trains or edits predictions."""
import copy
import hashlib
import io
import json
import math
import unicodedata
import uuid
import zipfile

import cv2
import numpy as np
from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, send_file, send_from_directory, session, url_for

from .evaluation import evaluate
from .models import db, GroundTruthSample, DetectionEvaluation, ImageAnalysis, Video, ImageReviewExample, now
from .presets import PRESETS
from .reviews import reviewer_name
from .routes import can_access, visitor_key
from .vision import LABELS

bp = Blueprint('dataset', __name__)
SPLITS = {'train': 'Оқыту', 'val': 'Валидация', 'test': 'Тәуелсіз сынақ'}


def scoped(model):
    return model.query if session.get('admin') else model.query.filter_by(owner_key=visitor_key())


def accessible(model, identifier):
    record = db.session.get(model, identifier)
    if not record or not can_access(record):
        abort(404)
    return record


def folder():
    path = current_app.config['STORAGE'] / 'ground_truth'
    path.mkdir(exist_ok=True, mode=0o700)
    return path


def normalized_group(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def components(samples):
    """Connected components keep video, identical image and road groups together."""
    parents = {s.id: s.id for s in samples}
    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    seen = {}
    for sample in samples:
        keys = [sample.source_group, 'pixels:' + sample.image_hash]
        if sample.road_group:
            keys.append('road:' + normalized_group(sample.road_group))
        for key in keys:
            if key in seen:
                parents[find(sample.id)] = find(seen[key])
            else:
                seen[key] = sample.id
    return {sample.id: find(sample.id) for sample in samples}


def checked_dataset():
    # Drafts participate in grouping too, so a shared road cannot bridge splits later.
    samples = scoped(GroundTruthSample).order_by(GroundTruthSample.id).all()
    groups = components(samples)
    split_by_group = {}
    for sample in samples:
        if sample.boxes is None:
            continue
        if sample.split not in SPLITS:
            raise ValueError('Жинақ бөлімі жарамсыз.')
        previous = split_by_group.setdefault(groups[sample.id], sample.split)
        if previous != sample.split:
            raise ValueError('Бір бейне немесе учаске бірнеше бөлімге түскен. Топтың бөлімін қайта сақтаңыз.')
    return [s for s in samples if s.boxes is not None], groups


def validated_boxes(value, width, height):
    if not isinstance(value, list) or len(value) > 200:
        raise ValueError('Ең көбі 200 қорап сақтаңыз.')
    boxes = []
    for item in value:
        if not isinstance(item, dict) or item.get('category') not in LABELS:
            raise ValueError('Ақау санатын таңдаңыз.')
        box = item.get('bbox')
        if not isinstance(box, list) or len(box) != 4 or any(
                isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) for n in box):
            raise ValueError('Қорап координаттары төрт ақырлы сан болуы тиіс.')
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Қорап сурет шекарасында және өлшемі нөлден үлкен болуы тиіс.')
        boxes.append(dict(category=item['category'], bbox=list(box)))
    return boxes


def original(sample):
    content = (folder() / f'{sample.id}.png').read_bytes()
    if hashlib.sha256(content).hexdigest() != sample.image_hash:
        raise ValueError('Бастапқы кадр өзгерген. Бағалау тоқтатылды.')
    return content


def create_sample(content, owner, source_key, source_group, **source):
    existing = GroundTruthSample.query.filter_by(owner_key=owner, source_key=source_key).first()
    if existing:
        return existing
    frame = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('Бастапқы кадр ашылмады.')
    identifier = uuid.uuid4().hex
    path = folder() / f'{identifier}.png'
    try:
        with path.open('xb') as destination:
            destination.write(content)
        path.chmod(0o600)
        sample = GroundTruthSample(id=identifier, owner_key=owner, source_key=source_key,
                    source_group=source_group, width=frame.shape[1], height=frame.shape[0],
                    image_hash=hashlib.sha256(content).hexdigest(), **source)
        db.session.add(sample)
        db.session.flush()
        members = GroundTruthSample.query.filter_by(owner_key=owner).all()
        groups = components(members)
        related = [member for member in members if member.id != identifier and groups[member.id] == groups[identifier]]
        if related:
            sample.split = related[0].split
        db.session.commit()
        return sample
    except Exception:
        db.session.rollback()
        path.unlink(missing_ok=True)
        raise


@bp.get('/evaluation')
def index():
    samples = scoped(GroundTruthSample).order_by(GroundTruthSample.updated_at.desc()).all()
    return render_template('evaluation.html', samples=samples, splits=SPLITS,
                           runs=scoped(DetectionEvaluation).order_by(DetectionEvaluation.created_at.desc()).limit(10).all())


@bp.post('/evaluation/images/<image_id>')
def from_image(image_id):
    image = accessible(ImageAnalysis, image_id)
    content = (current_app.config['STORAGE'] / 'images' / f'{image.id}-original.png').read_bytes()
    key = 'image:' + hashlib.sha256(content).hexdigest()
    sample = create_sample(content, image.owner_key, key, key, image_id=image.id)
    return redirect(url_for('dataset.editor', sample_id=sample.id))


@bp.post('/evaluation/videos/<video_id>')
def from_video(video_id):
    video = accessible(Video, video_id)
    try:
        index = int(request.form.get('frame_index', ''))
        if not 0 <= index < round(video.duration * video.fps):
            raise ValueError()
    except ValueError:
        flash('Бейне шекарасындағы бүтін кадр нөмірін енгізіңіз.', 'error')
        return redirect(url_for('web.video_detail', video_id=video.id))
    path = current_app.config['STORAGE'] / 'raw' / video.stored_name
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        abort(400)
    ok, encoded = cv2.imencode('.png', frame)
    if not ok:
        abort(500)
    with path.open('rb') as raw:
        source_hash = hashlib.file_digest(raw, 'sha256').hexdigest()
    group = 'video:' + source_hash
    sample = create_sample(encoded.tobytes(), video.owner_key, f'{group}:{index}', group,
                           video_id=video.id, frame_index=index)
    return redirect(url_for('dataset.editor', sample_id=sample.id))


@bp.route('/evaluation/samples/<sample_id>', methods=['GET', 'POST'])
def editor(sample_id):
    sample = accessible(GroundTruthSample, sample_id)
    if request.method == 'POST':
        try:
            if request.form.get('complete') != 'yes':
                raise ValueError('Кадрдағы барлық төрт санаттағы көрінетін ақауды тексергеніңізді растаңыз.')
            payload = request.form.get('boxes', '')
            if len(payload) > 100000:
                raise ValueError('Таңба деректері тым үлкен.')
            try:
                parsed = json.loads(payload)
            except (ValueError, TypeError) as exc:
                raise ValueError('Қорап деректері жарамсыз. Бетті жаңартып, қайта белгілеңіз.') from exc
            boxes = validated_boxes(parsed, sample.width, sample.height)
            negative = request.form.get('negative') == 'yes'
            if (not boxes) != negative:
                raise ValueError('Қораптар жоқ болса «Ақау жоқ» деп растаңыз; теріс кадрда қорап болмауы тиіс.')
            road = normalized_group(request.form.get('road_group', ''))
            if len(road) > 120 or (sample.image_id and not road):
                raise ValueError('Фотосуретке учаске/сапар тобын көрсетіңіз (120 таңбаға дейін). Бір сапарға бір атау қолданыңыз.')
            split = request.form.get('split')
            if split not in SPLITS:
                raise ValueError('Жинақ бөлімін таңдаңыз.')
            try:
                revision = int(request.form.get('revision', '-1'))
            except ValueError as exc:
                raise ValueError('Таңба нұсқасы жарамсыз. Бетті жаңартыңыз.') from exc
            result = db.session.execute(db.update(GroundTruthSample).where(
                GroundTruthSample.id == sample.id, GroundTruthSample.revision == revision).values(
                boxes=boxes, negative=negative, road_group=road, split=split,
                revision=revision + 1, reviewer=reviewer_name(), updated_at=now()))
            if result.rowcount != 1:
                raise ValueError('Таңба басқа бетте өзгерген. Бетті жаңартып, қайта тексеріңіз.')
            db.session.expire_all()
            # A split belongs to the entire connected source/road group, never an individual frame.
            members = scoped(GroundTruthSample).all()
            groups = components(members)
            for member in members:
                if groups[member.id] == groups[sample.id]:
                    if member.id != sample.id and member.split != split:
                        member.revision += 1
                        member.updated_at = now()
                    member.split = split
            db.session.commit()
            flash('Эталондық таңбалар сақталды. Байланысқан топтың бөлімі бірге жаңартылды.', 'success')
            return redirect(url_for('dataset.editor', sample_id=sample.id))
        except (ValueError, TypeError) as exc:
            db.session.rollback()
            flash(str(exc), 'error')
            return render_editor(sample), 400
    return render_editor(sample)


def render_editor(sample):
    image = db.session.get(ImageAnalysis, sample.image_id) if sample.image_id else None
    video = db.session.get(Video, sample.video_id) if sample.video_id else None
    reviews = {}
    if image:
        for review in ImageReviewExample.query.filter_by(image_id=image.id).order_by(ImageReviewExample.created_at.desc()).all():
            reviews.setdefault(review.prediction_index, review)
    return render_template('annotation.html', sample=sample, splits=SPLITS, image=image, video=video, reviews=reviews)


@bp.get('/evaluation/samples/<sample_id>/image.png')
def sample_image(sample_id):
    accessible(GroundTruthSample, sample_id)
    return send_from_directory(folder(), f'{sample_id}.png', mimetype='image/png')


def manifest(sample, groups):
    return dict(id=sample.id, source_key=sample.source_key, source_group=sample.source_group,
                road_group=sample.road_group, group=groups[sample.id], split=sample.split,
                image_hash=sample.image_hash, width=sample.width, height=sample.height,
                truth=copy.deepcopy(sample.boxes), negative=sample.negative, revision=sample.revision,
                reviewer=sample.reviewer, frame_index=sample.frame_index)


@bp.get('/evaluation/export')
def export_dataset():
    try:
        samples, groups = checked_dataset()
        if not samples:
            raise ValueError('Алдымен кемінде бір кадрды толық таңбалаңыз.')
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for split in SPLITS:
                for kind in ('images', 'labels'):
                    archive.writestr(f'{kind}/{split}/', '')
            for sample in samples:
                lines = []
                for item in sample.boxes:
                    x1, y1, x2, y2 = item['bbox']
                    lines.append(f"{list(LABELS).index(item['category'])} {(x1+x2)/2/sample.width:.12g} "
                                 f"{(y1+y2)/2/sample.height:.12g} {(x2-x1)/sample.width:.12g} {(y2-y1)/sample.height:.12g}")
                archive.writestr(f'images/{sample.split}/{sample.id}.png', original(sample))
                archive.writestr(f'labels/{sample.split}/{sample.id}.txt', '\n'.join(lines))
            archive.writestr('data.yaml', 'train: images/train\nval: images/val\ntest: images/test\nnames:\n' +
                             ''.join(f'  {index}: {key}\n' for index, key in enumerate(LABELS)))
            archive.writestr('manifest.json', json.dumps([manifest(s, groups) for s in samples], ensure_ascii=False, indent=2))
            archive.writestr('README.txt', 'Тек адам толық тексерген кадрлар. Бос TXT — расталған теріс кадр.\n'
                'Бір бейне, бірдей PNG немесе бір учаске тобы бір бөлімде сақталады.\n'
                'Әртүрлі файлдағы жақын көшірмелер мен бір жолдың сапарларын ортақ учаске атауымен байланыстырыңыз.\n'
                'Таңбалар адамның шешімі; тәуелсіз сараптамамен тексеру ұсынылады. Оқыту іске қосылмайды.\n')
        buffer.seek(0)
        return send_file(buffer, mimetype='application/zip', as_attachment=True, download_name='JolBaqylau-ground-truth.zip')
    except (ValueError, OSError) as exc:
        flash(f'Экспорт орындалмады: {exc}', 'error')
        return redirect(url_for('dataset.index'))


@bp.post('/evaluation/compare')
def compare():
    try:
        samples, groups = checked_dataset()
        validation = [s for s in samples if s.split == 'val']
        if not validation:
            raise ValueError('Толық таңбаланған валидация кадры жоқ. Топқа «Валидация» бөлімін таңдаңыз.')
        if len(validation) > 32:
            raise ValueError('Шағын тексеруге ең көбі 32 валидация кадрын қалдырыңыз.')
        records = [manifest(sample, groups) for sample in validation]
        settings = {key: copy.deepcopy(PRESETS[key]) for key in ('high_recall', 'balanced')}
        predictions = {key: [] for key in settings}
        detector = current_app.extensions['road_detector']
        for sample, record in zip(validation, records):
            frame = cv2.imdecode(np.frombuffer(original(sample), np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('Валидация кадры ашылмады.')
            for key, config in settings.items():
                output = detector.predict(frame, config['confidence'], imgsz=config['inference_size'])
                predictions[key].append(dict(sample_id=sample.id, truth=record['truth'], predictions=output))
        result = dict(scope='frame_detection_without_temporal_filter', iou=.5, dataset=records,
                      model=current_app.config['MODEL_REPO'], revision=current_app.config['MODEL_REVISION'],
                      presets=settings, predictions=predictions,
                      metrics={key: evaluate(rows, .5) for key, rows in predictions.items()})
        run = DetectionEvaluation(id=uuid.uuid4().hex, owner_key=None if session.get('admin') else visitor_key(), snapshot=result)
        db.session.add(run)
        db.session.commit()
        return redirect(url_for('dataset.report', run_id=run.id))
    except ValueError as exc:
        db.session.rollback()
        flash(str(exc), 'error')
        return redirect(url_for('dataset.index'))
    except Exception:
        db.session.rollback()
        current_app.logger.exception('Пресеттерді бағалау сәтсіз')
        flash('Бағалау орындалмады. Модель мен бастапқы кадрларды тексеріңіз; жартылай метрика сақталмады.', 'error')
        return render_template('error.html', message='Бағалау орындалмады. Сервер журналын тексеріңіз.'), 503


@bp.get('/evaluation/reports/<run_id>')
def report(run_id):
    return render_template('evaluation_report.html', run=accessible(DetectionEvaluation, run_id))


@bp.get('/evaluation/reports/<run_id>.json')
def report_json(run_id):
    run = accessible(DetectionEvaluation, run_id)
    return send_file(io.BytesIO(json.dumps(run.snapshot, ensure_ascii=False, indent=2).encode()),
                     mimetype='application/json', as_attachment=True, download_name='JolBaqylau-evaluation.json')
