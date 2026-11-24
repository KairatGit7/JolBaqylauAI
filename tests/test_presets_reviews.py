import io
import json
from pathlib import Path
import zipfile

import pytest

from jol.models import db, Video, ImageAnalysis, DetectionEvent, ReviewExample, ImageReviewExample, migrate_early_schema
from test_public_images import image_bytes, token


@pytest.mark.parametrize('preset,confidence,size', [('default', .6, 640), ('high_recall', .15, 640), ('balanced', .4, 960)])
def test_image_presets_are_explicit_and_never_temporal(app, monkeypatch, preset, confidence, size):
    calls = []
    def predict(frame, threshold, **kwargs):
        calls.append((threshold, kwargs.get('imgsz', 640)))
        return [dict(category='pothole', confidence=.8, bbox=[10, 20, 50, 45])]
    monkeypatch.setattr(app.extensions['road_detector'], 'predict', predict)
    visitor = app.test_client()
    response = visitor.post('/images', data={'csrf_token': token(visitor), 'preset': preset,
        'confidence': '.99', 'image': (io.BytesIO(image_bytes()), 'photo.png')})
    assert response.status_code == 302
    assert calls == [(confidence, size)]
    with app.app_context():
        result = ImageAnalysis.query.one()
        assert (result.preset, result.confidence, result.inference_size) == (preset, confidence, size)
        assert len(result.predictions) == 1  # Balanced must not suppress a single-image detection.
        assert Video.query.count() == 0
    page = visitor.get(response.location).data.decode()
    assert {'default': 'Ағымдағы баптаулар', 'high_recall': 'Толық анықтау', 'balanced': 'Теңгерімді'}[preset] in page
    assert 'уақыт сүзгісі қолданылмайды' in page
    for label in ['Дұрыс анықталды', 'Қате анықталды', 'Тексерілмеген']:
        assert label in page
    assert app.config['MODEL_CONFIDENCE'] == .6 and app.config['SAMPLE_SECONDS'] == 1


@pytest.mark.parametrize('preset,confidence,size,min_hits,frames', [
    ('default', .6, 640, 2, 2), ('high_recall', .15, 640, 1, 20), ('balanced', .4, 960, 2, 20)])
def test_video_presets_frame_sampling_and_consecutive_filter(app, video_bytes, monkeypatch,
                                                           preset, confidence, size, min_hits, frames):
    visitor = app.test_client()
    response = visitor.post('/analysis', data={'csrf_token': token(visitor, '/analysis'), 'preset': preset,
        'observed_at': '2026-01-01T12:00', 'video': (io.BytesIO(video_bytes), 'road.avi')})
    identifier = response.location.rsplit('/', 1)[-1]
    assert response.status_code == 302
    calls = []
    def predict(frame, threshold, **kwargs):
        index = len(calls)
        calls.append((threshold, kwargs.get('imgsz', 640)))
        # Explicit unit-test predictions: one unstable candidate, then two consecutive hits.
        if index in (0, 2, 3):
            return [dict(category='pothole', confidence=.8, bbox=[10, 20, 50, 45])]
        return []
    monkeypatch.setattr(app.extensions['road_detector'], 'load', lambda: None)
    monkeypatch.setattr(app.extensions['road_detector'], 'predict', predict)
    app.extensions['job_runner'].process(identifier)
    assert calls == [(confidence, size)] * frames
    with app.app_context():
        video = db.session.get(Video, identifier)
        assert video.job.status == 'completed', video.job.error
        assert video.job.frames_analyzed == frames
        assert (video.preset, video.confidence, video.inference_size, video.temporal_min_hits) == (preset, confidence, size, min_hits)
        assert video.sample_seconds == (1 if preset == 'default' else 1 / video.fps)
        assert video.temporal_max_gap == pytest.approx(video.sample_seconds * 1.1)
        events = sorted(video.events, key=lambda event: event.start_seconds)
        assert events[0].filter_status == ('passed' if preset == 'high_recall' else 'suppressed')
        if preset != 'default':
            assert len(events) == 2 and events[1].filter_status == 'passed' and len(events[1].hits) == 2
        assert all(event.review_status == 'predicted' for event in events)
    assert visitor.get(response.location).status_code == 200


def test_default_configuration_and_preset_deduplication(app, video_bytes):
    app.config.update(MODEL_CONFIDENCE=.72, SAMPLE_SECONDS=.5, TEMPORAL_MIN_HITS=3)
    visitor = app.test_client()
    csrf = token(visitor, '/analysis')
    def upload(preset=None):
        data = dict(csrf_token=csrf, observed_at='2026-01-01T12:00', video=(io.BytesIO(video_bytes), 'road.avi'))
        if preset:
            data['preset'] = preset
        return visitor.post('/analysis', data=data)
    default = upload()
    high = upload('high_recall')
    balanced = upload('balanced')
    assert len({default.location, high.location, balanced.location}) == 3
    assert upload('high_recall').location == high.location
    with app.app_context():
        video = db.session.get(Video, default.location.rsplit('/', 1)[-1])
        assert (video.confidence, video.sample_seconds, video.temporal_min_hits, video.inference_size) == (.72, .5, 3, 640)
    assert app.config['MODEL_CONFIDENCE'] == .72
    # Reanalysis uses the same selector and settings-aware cache, without mutating old results.
    response = visitor.post(default.location + '/reanalyze', data={'csrf_token': csrf, 'preset': 'balanced'})
    assert response.location == balanced.location


def test_invalid_presets_rejected_and_default_selected(app, video_bytes, monkeypatch):
    visitor = app.test_client()
    csrf = token(visitor)
    assert 'value="default"  selected' in visitor.get('/images').data.decode()
    for route, field, payload, name in [('/images', 'image', image_bytes(), 'photo.png'),
                                       ('/analysis', 'video', video_bytes, 'road.avi')]:
        response = visitor.post(route, data={'csrf_token': csrf, 'preset': 'unknown',
            'observed_at': '2026-01-01T12:00', field: (io.BytesIO(payload), name)})
        assert response.status_code == 400
    with app.app_context():
        assert ImageAnalysis.query.count() == 0 and Video.query.count() == 0


def test_image_reviews_storage_export_retraction_and_access(app, client, monkeypatch):
    predictions = [dict(category='pothole', confidence=.8, bbox=[10, 20, 50, 45]),
                   dict(category='pothole', confidence=.7, bbox=[50, 20, 75, 45])]
    monkeypatch.setattr(app.extensions['road_detector'], 'predict', lambda *args, **kwargs: predictions)
    visitor = app.test_client()
    csrf = token(visitor)
    response = visitor.post('/images', data={'csrf_token': csrf, 'preset': 'balanced',
        'image': (io.BytesIO(image_bytes()), 'photo.png')})
    image_id = response.location.rsplit('/', 1)[-1]
    endpoint = response.location + '/predictions/0/review'
    assert visitor.post(endpoint, data={'csrf_token': csrf, 'verdict': 'confirmed', 'note': ''}).status_code == 302
    with app.app_context():
        assert ImageReviewExample.query.count() == 0
    visitor.post(endpoint, data={'csrf_token': csrf, 'verdict': 'rejected', 'reason': 'manhole_cover',
                                'note': 'Synthetic test review, not experimental ground truth.'})
    second_endpoint = response.location + '/predictions/1/review'
    visitor.post(second_endpoint, data={'csrf_token': csrf, 'verdict': 'confirmed', 'note': 'Synthetic second label.'})
    with app.app_context():
        assert db.session.get(ImageAnalysis, image_id).predictions == predictions
        examples = ImageReviewExample.query.all()
        first_id = next(row.id for row in examples if row.prediction_index == 0)
        assert len(examples) == 2
        assert all(row.snapshot['preset'] == 'balanced' and row.snapshot['inference_size'] == 960 for row in examples)
        assert all(row.snapshot['exhaustively_labeled'] is False for row in examples)
        assert all(row.reviewer.startswith('session-') for row in examples)
    assert visitor.post(endpoint, data={'verdict': 'predicted'}).status_code == 400
    stranger = app.test_client()
    stranger_csrf = token(stranger)
    assert stranger.post(endpoint, data={'csrf_token': stranger_csrf, 'verdict': 'predicted'}).status_code == 404
    assert visitor.get('/reviews/export').status_code == 302  # Whole-database export stays administrative.
    exported = zipfile.ZipFile(io.BytesIO(client.get('/reviews/export').data))
    records = [json.loads(line) for line in exported.read('reviews.jsonl').splitlines()]
    assert len(records) == 2 and {row['verdict'] for row in records} == {'confirmed', 'rejected'}
    assert all(exported.read(row['image']).startswith(b'\x89PNG') for row in records)
    assert client.get('/reviews').status_code == 200
    assert client.get(f'/reviews/{first_id}/image').status_code == 200
    visitor.post(endpoint, data={'csrf_token': csrf, 'verdict': 'predicted'})
    exported = zipfile.ZipFile(io.BytesIO(client.get('/reviews/export').data))
    records = [json.loads(line) for line in exported.read('reviews.jsonl').splitlines()]
    assert len(records) == 1 and records[0]['prediction_index'] == 1
    with app.app_context():
        assert ImageReviewExample.query.count() == 3
        assert db.session.get(ImageReviewExample, first_id).verdict == 'rejected'


def test_owner_video_reviews_and_sqlite_history(app, video_bytes, monkeypatch):
    visitor = app.test_client()
    csrf = token(visitor, '/analysis')
    response = visitor.post('/analysis', data={'csrf_token': csrf, 'preset': 'high_recall',
        'observed_at': '2026-01-01T12:00', 'video': (io.BytesIO(video_bytes), 'road.avi')})
    identifier = response.location.rsplit('/', 1)[-1]
    monkeypatch.setattr(app.extensions['road_detector'], 'load', lambda: None)
    monkeypatch.setattr(app.extensions['road_detector'], 'predict', lambda *args, **kwargs:
                        [dict(category='pothole', confidence=.8, bbox=[10, 20, 50, 45])])
    app.extensions['job_runner'].process(identifier)
    with app.app_context():
        event_id = DetectionEvent.query.one().id
    endpoint = f'/events/{event_id}/review'
    stranger = app.test_client()
    assert stranger.post(endpoint, data={'csrf_token': token(stranger), 'verdict': 'confirmed', 'note': 'x'}).status_code == 302
    for verdict in ['confirmed', 'rejected', 'predicted']:
        result = visitor.post(endpoint, data={'csrf_token': csrf, 'verdict': verdict, 'reason': 'shadow',
                                             'note': 'Synthetic review storage test.'})
        assert result.status_code == 302
        with app.app_context():
            assert db.session.get(DetectionEvent, event_id).review_status == verdict
            assert db.session.get(DetectionEvent, event_id).category == 'pothole'
    with app.app_context():
        assert ReviewExample.query.count() == 3
        assert all(row.snapshot['preset'] == 'high_recall' for row in ReviewExample.query.all())


def test_additive_preset_migration(app):
    with app.app_context():
        for table in ['video', 'image_analysis']:
            for column in ['preset', 'inference_size']:
                db.session.execute(db.text(f'ALTER TABLE {table} DROP COLUMN {column}'))
        db.session.commit()
        migrate_early_schema()
        migrate_early_schema()
        inspector = db.inspect(db.engine)
        for table in ['video', 'image_analysis']:
            assert {'preset', 'inference_size'} <= {column['name'] for column in inspector.get_columns(table)}


@pytest.mark.integration
def test_real_balanced_image_preserves_single_image_detection(app):
    visitor = app.test_client()
    path = Path(__file__).resolve().parents[1] / 'samples/rdd2022/China_Drone_000101.jpg'
    result = visitor.post('/images', data={'csrf_token': token(visitor), 'preset': 'balanced',
        'image': (io.BytesIO(path.read_bytes()), path.name)})
    assert result.status_code == 302
    with app.app_context():
        image = ImageAnalysis.query.one()
        assert image.inference_size == 960 and image.confidence == .4
        assert any(prediction['category'] == 'pothole' for prediction in image.predictions)
