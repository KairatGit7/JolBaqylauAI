"""Private review archive. No automatic conversion to exhaustive training labels."""
import hashlib
import io
import json
import uuid
import zipfile

import cv2
from flask import Blueprint, abort, current_app, render_template, send_file, send_from_directory, session

from .models import db, ReviewExample, ImageReviewExample
from .routes import login_required

bp = Blueprint('reviews', __name__)
REASONS = {
    'manhole_cover': 'Люк қақпағы', 'road_marking': 'Жол таңбасы', 'shadow': 'Көлеңке',
    'asphalt_patch': 'Асфальт жамауы', 'duplicate': 'Қайталанған белгі',
    'wrong_category': 'Санаты қате', 'other': 'Басқа себеп',
}


def reviewer_name():
    if session.get('admin'):
        return 'local-admin'
    # Never export the browser's access credential.
    return 'session-' + hashlib.sha256(session['visitor_key'].encode()).hexdigest()[:16]


def archive_review(event, verdict, note, reason):
    if reason not in REASONS:
        raise ValueError('Қате анықтау себебін тізімнен таңдаңыз.')
    video = event.video
    hit = max(event.hits, key=lambda item: item['confidence'])
    frame_index = round(hit['timestamp'] * video.fps)
    cap = cv2.VideoCapture(str(current_app.config['STORAGE'] / 'raw' / video.stored_name))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise OSError('Бастапқы бейне кадры ашылмады. Түпнұсқа файлды тексеріңіз.')
    ok, encoded = cv2.imencode('.png', frame)
    if not ok:
        raise OSError('Бастапқы кадрды сақтау мүмкін болмады.')
    identifier = uuid.uuid4().hex
    image_name = identifier + '.png'
    image_path = current_app.config['STORAGE'] / 'reviewed' / image_name
    contents = encoded.tobytes()
    snapshot = dict(version=1, event_id=event.id, video_id=video.id, video_sha256=video.sha256,
                    source_type='video', preset=video.preset, inference_size=video.inference_size,
                    split_group=video.sha256 or video.id, model_revision=video.model_revision,
                    confidence_threshold=video.confidence, pipeline_version=video.pipeline_version,
                    temporal_min_hits=video.temporal_min_hits, temporal_iou=video.temporal_iou,
                    temporal_max_gap=video.temporal_max_gap, filter_status=event.filter_status,
                    category=event.category, confidence=event.confidence, bbox=list(event.bbox),
                    timestamp=hit['timestamp'], frame_index=frame_index, width=video.width, height=video.height,
                    hits=event.hits, is_demo=video.is_demo, label_scope='prediction_box_only',
                    exhaustively_labeled=False, image_sha256=hashlib.sha256(contents).hexdigest())
    try:
        with image_path.open('xb') as destination:
            destination.write(contents)
        image_path.chmod(0o600)
    except OSError:
        image_path.unlink(missing_ok=True)
        raise
    return ReviewExample(id=identifier, event_id=event.id, verdict=verdict,
                         reason=reason if verdict == 'rejected' else ('reset' if verdict == 'predicted' else 'correct'),
                         note=note, reviewer=reviewer_name(), image_name=image_name, snapshot=snapshot), image_path


def archive_image_review(image, index, verdict, note, reason):
    if reason not in REASONS:
        raise ValueError('Қате анықтау себебін тізімнен таңдаңыз.')
    prediction = image.predictions[index]
    contents = (current_app.config['STORAGE'] / 'images' / f'{image.id}-original.png').read_bytes()
    image_hash = hashlib.sha256(contents).hexdigest()
    identifier = uuid.uuid4().hex
    name = identifier + '.png'
    path = current_app.config['STORAGE'] / 'reviewed' / name
    snapshot = dict(version=1, source_type='image', image_id=image.id, prediction_index=index,
                    category=prediction['category'], confidence=prediction['confidence'], bbox=prediction['bbox'],
                    width=image.width, height=image.height, preset=image.preset, inference_size=image.inference_size,
                    confidence_threshold=image.confidence, model_revision=image.model_revision,
                    split_group='image:' + image_hash, image_sha256=image_hash,
                    temporal_min_hits=None, label_scope='prediction_box_only', exhaustively_labeled=False)
    reviewer = reviewer_name()
    try:
        with path.open('xb') as destination:
            destination.write(contents)
        path.chmod(0o600)
    except OSError:
        path.unlink(missing_ok=True)
        raise
    return ImageReviewExample(id=identifier, image_id=image.id, prediction_index=index, verdict=verdict,
                              reason=reason if verdict == 'rejected' else ('reset' if verdict == 'predicted' else 'correct'),
                              note=note, reviewer=reviewer, image_name=name, snapshot=snapshot), path


def latest_reviews():
    latest = {}
    for example in ReviewExample.query.order_by(ReviewExample.created_at.desc(), ReviewExample.id.desc()).all():
        latest.setdefault(example.event_id, example)
    return list(latest.values())


def latest_image_reviews():
    latest = {}
    for example in ImageReviewExample.query.order_by(ImageReviewExample.created_at.desc(), ImageReviewExample.id.desc()).all():
        latest.setdefault((example.image_id, example.prediction_index), example)
    return list(latest.values())


@bp.get('/reviews')
@login_required
def reviews_page():
    return render_template('reviews.html', examples=latest_reviews() + latest_image_reviews(),
                           history_count=ReviewExample.query.count() + ImageReviewExample.query.count())


@bp.get('/reviews/<example_id>/image')
@login_required
def review_image(example_id):
    example = db.session.get(ReviewExample, example_id) or db.session.get(ImageReviewExample, example_id)
    if example is None:
        abort(404)
    return send_from_directory(current_app.config['STORAGE'] / 'reviewed', example.image_name)


@bp.get('/reviews/export')
@login_required
def export_reviews():
    buffer = io.BytesIO()
    records = []
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for example in latest_reviews() + latest_image_reviews():
            if example.verdict == 'predicted':
                continue  # A reset retracts the previous training/evaluation label.
            path = current_app.config['STORAGE'] / 'reviewed' / example.image_name
            name = 'images/' + example.image_name
            archive.write(path, name)
            records.append(dict(example.snapshot, review_id=example.id, verdict=example.verdict,
                                reason=example.reason, note=example.note, reviewer=example.reviewer,
                                reviewed_at=example.created_at.isoformat() + 'Z', image=name))
        archive.writestr('reviews.jsonl', '\n'.join(json.dumps(row, ensure_ascii=False) for row in records))
        archive.writestr('README.txt', 'Prediction-box review labels, NOT exhaustive image annotations.\n'
                         'Rejected boxes are not automatically background: inspect reason and relabel.\n'
                         'Only latest non-retracted decisions are exported. History remains in the database.\n'
                         'Group train/validation/test by split_group (source video SHA-256), never adjacent frames.\n'
                         'Image records group by original normalized PNG hash; inspect near-duplicates before splitting.\n'
                         'Annotate missed defects and adjudicate labels before evaluation or fine-tuning.\n')
    buffer.seek(0)
    return send_file(buffer, mimetype='application/zip', as_attachment=True, download_name='reviewed-examples.zip')
