import io
import json
import zipfile

import cv2
import pytest

from conftest import upload_video
from jol.models import db, Video, DetectionEvent, ReviewExample, migrate_early_schema
from jol.vision import EventAggregator, suppress_duplicates


def detection(confidence=.8, bbox=None, category='pothole'):
    return dict(category=category, confidence=confidence, bbox=bbox or [0, 0, 50, 50])


def test_threshold_snapshot_and_settings_aware_dedup(app, client, video_bytes):
    first = upload_video(client, video_bytes)
    with app.app_context():
        original = Video.query.one()
        assert original.confidence == .6
        assert original.temporal_min_hits == 2
        assert original.pipeline_version == 'temporal-v1'
        assert original.temporal_max_gap == pytest.approx(1.1)
        original.job.status = 'completed'
        original.is_current = True
        db.session.commit()
        identifier = original.id
    assert upload_video(client, video_bytes).location == first.location
    rerun = client.post(f'/videos/{identifier}/reanalyze', data={'csrf_token': 'test-csrf', 'confidence': '.75'})
    assert rerun.status_code == 302 and rerun.location != first.location
    assert client.post(f'/videos/{identifier}/reanalyze', data={'csrf_token': 'test-csrf', 'confidence': '.75'}).location == rerun.location
    with app.app_context():
        original = db.session.get(Video, identifier)
        assert original.confidence == .6 and original.job.status == 'completed'
        assert Video.query.count() == 2
        assert db.session.get(Video, rerun.location.rsplit('/', 1)[-1]).stored_name == original.stored_name
    for invalid in ('nan', 'inf', '0', '1.1', 'hello'):
        response = client.post('/analysis', data={'csrf_token': 'test-csrf', 'confidence': invalid,
            'observed_at': '2026-01-01T12:00', 'video': (io.BytesIO(video_bytes), 'test.avi')})
        assert response.status_code == 400
    with app.app_context():
        assert Video.query.count() == 2


def test_temporal_stability_class_and_empty_frame_boundaries():
    tracker = EventAggregator(max_gap=1.1, min_iou=.2, min_hits=2, duplicate_iou=.5)
    tracker.add([detection(), detection(.7, [1, 1, 51, 51]), detection(bbox=[80, 80, 100, 100])], 0, 'a', 'v')
    tracker.add([detection(.9, [2, 2, 52, 52])], 1, 'b', 'v')
    tracker.add([], 2, 'c', 'v')
    tracker.add([detection()], 3, 'd', 'v')
    tracker.add([detection(category='transverse_crack')], 4, 'e', 'v')
    result = list(tracker.results())
    assert len(result) == 4
    assert [row['filter_status'] for row in result] == ['passed', 'suppressed', 'suppressed', 'suppressed']
    assert result[0]['confidence'] == .9 and result[0]['evidence_name'] == 'b'
    assert len(result[0]['hits']) == 2
    with pytest.raises(ValueError):
        tracker.add([detection()], 4, 'e', 'v')
    # Empty frames cannot be bridged even with an unusually generous time gap.
    tracker = EventAggregator(max_gap=10, min_hits=2)
    tracker.add([detection()], 0, 'a', 'v')
    tracker.add([], .5, 'b', 'v')
    tracker.add([detection()], 1, 'c', 'v')
    assert all(row['filter_status'] == 'suppressed' for row in tracker.results())


def test_nms_retains_different_classes_and_disjoint_defects():
    result = suppress_duplicates([detection(), detection(.7), detection(category='transverse_crack'),
                                  detection(bbox=[80, 80, 100, 100])])
    assert len(result) == 3


def test_high_confidence_persistent_false_positive_is_not_semantically_rejected():
    tracker = EventAggregator(max_gap=1.1, min_hits=2)
    for timestamp in (0, 1):
        tracker.add([detection(.95)], timestamp, 'a', 'v')
    assert list(tracker.results())[0]['filter_status'] == 'passed'
    # Stability never implies human confirmation or manhole recognition.
    assert 'review_status' not in list(tracker.results())[0]


@pytest.fixture
def processed(app, client, video_bytes, monkeypatch):
    response = upload_video(client, video_bytes)
    identifier = response.location.rsplit('/', 1)[-1]
    detector = app.extensions['road_detector']
    monkeypatch.setattr(detector, 'load', lambda: None)
    predictions = iter([[detection(), detection(bbox=[80, 60, 130, 110])], [detection()]])
    monkeypatch.setattr(detector, 'predict', lambda frame, confidence: next(predictions))
    app.extensions['job_runner'].process(identifier)
    with app.app_context():
        video = db.session.get(Video, identifier)
        assert video.job.status == 'completed'
        return identifier, {event.filter_status: event.id for event in video.events}


def test_worker_suppression_and_human_rescue(app, client, processed):
    identifier, ids = processed
    assert set(ids) == {'passed', 'suppressed'}
    assert [event['id'] for event in client.get('/api/events').json] == [ids['passed']]
    page = client.get(f'/videos/{identifier}').data.decode()
    assert 'Дұрыс анықталды' in page and 'Қате анықталды' in page and 'тұрақсыз кандидат' in page
    client.post(f'/events/{ids["suppressed"]}/review', data={'csrf_token': 'test-csrf',
                'verdict': 'confirmed', 'note': 'Synthetic fixture: reviewer rescue.'})
    assert len(client.get('/api/events').json) == 2
    assert len(client.get('/api/events?status=confirmed').json) == 1
    with app.app_context():
        event = db.session.get(DetectionEvent, ids['suppressed'])
        assert event.filter_status == 'suppressed' and event.review_status == 'confirmed'
        assert ReviewExample.query.one().verdict == 'confirmed'


def test_review_archive_versions_and_retraction(app, client, processed):
    identifier, ids = processed
    event_id = ids['passed']
    endpoint = f'/events/{event_id}/review'
    client.post(endpoint, data={'csrf_token': 'test-csrf', 'verdict': 'rejected', 'reason': 'manhole_cover',
                               'note': 'Synthetic test label; not an experimental finding.'})
    with app.app_context():
        example = ReviewExample.query.one()
        first_id = example.id
        assert example.snapshot['label_scope'] == 'prediction_box_only'
        assert example.snapshot['exhaustively_labeled'] is False
        assert example.snapshot['confidence_threshold'] == .6
        assert example.snapshot['split_group'] == db.session.get(Video, identifier).sha256
        assert cv2.imread(str(app.config['STORAGE'] / 'reviewed' / example.image_name)).shape == (120, 160, 3)
        assert db.session.get(DetectionEvent, event_id).category == 'pothole'
    archive = zipfile.ZipFile(io.BytesIO(client.get('/reviews/export').data))
    records = [json.loads(line) for line in archive.read('reviews.jsonl').splitlines()]
    assert len(records) == 1 and records[0]['reason'] == 'manhole_cover'
    assert archive.read(records[0]['image']).startswith(b'\x89PNG')
    client.post(endpoint, data={'csrf_token': 'test-csrf', 'status': 'predicted', 'note': ''})
    archive = zipfile.ZipFile(io.BytesIO(client.get('/reviews/export').data))
    assert archive.read('reviews.jsonl') == b''
    with app.app_context():
        assert ReviewExample.query.count() == 2
        assert db.session.get(ReviewExample, first_id).verdict == 'rejected'
    assert client.get('/reviews').status_code == 200
    assert client.get(f'/reviews/{first_id}/image').headers['Cache-Control'] == 'private, no-store'
    anonymous = app.test_client()
    for path in ['/reviews', '/reviews/export', f'/reviews/{first_id}/image']:
        assert anonymous.get(path).status_code == 302
    assert client.post(endpoint, data={'verdict': 'confirmed'}).status_code == 400


def test_review_failure_does_not_change_prediction(app, client, processed, monkeypatch):
    from jol import reviews
    identifier, ids = processed
    def fail(*args):
        raise OSError('unavailable original')
    monkeypatch.setattr(reviews, 'archive_review', fail)
    response = client.post(f'/events/{ids["passed"]}/review', data={'csrf_token': 'test-csrf',
                 'verdict': 'rejected', 'reason': 'shadow', 'note': 'Test failure.'}, follow_redirects=True)
    assert 'Тексеру сақталмады' in response.data.decode()
    with app.app_context():
        assert ReviewExample.query.count() == 0
        assert db.session.get(DetectionEvent, ids['passed']).review_status == 'predicted'


def test_new_success_supersedes_counts_but_failure_keeps_original(app, client, processed, monkeypatch):
    identifier, ids = processed
    response = client.post(f'/videos/{identifier}/reanalyze', data={'csrf_token': 'test-csrf', 'confidence': '.75'})
    new_id = response.location.rsplit('/', 1)[-1]
    detector = app.extensions['road_detector']
    def fail(frame, confidence):
        raise RuntimeError('inference failure')
    monkeypatch.setattr(detector, 'predict', fail)
    app.extensions['job_runner'].process(new_id)
    assert [event['id'] for event in client.get('/api/events').json] == [ids['passed']]
    response = client.post(f'/videos/{identifier}/reanalyze', data={'csrf_token': 'test-csrf', 'confidence': '.75'})
    successful_id = response.location.rsplit('/', 1)[-1]
    monkeypatch.setattr(detector, 'predict', lambda frame, confidence: [detection()])
    app.extensions['job_runner'].process(successful_id)
    assert len(client.get('/api/events').json) == 1
    with app.app_context():
        assert db.session.get(Video, identifier).is_current is False
        assert db.session.get(Video, successful_id).is_current is True
        assert len(db.session.get(Video, identifier).events) == 2
    assert client.get(f'/videos/{identifier}').status_code == 200


def test_review_commit_failure_cleans_archive(app, client, processed, monkeypatch):
    from sqlalchemy.exc import SQLAlchemyError
    _, ids = processed
    def fail():
        raise SQLAlchemyError('test transaction failure')
    monkeypatch.setattr(db.session, 'commit', fail)
    response = client.post(f'/events/{ids["passed"]}/review', data={'csrf_token': 'test-csrf',
                 'verdict': 'rejected', 'reason': 'shadow', 'note': 'Test transaction failure.'})
    assert response.status_code == 302
    assert list((app.config['STORAGE'] / 'reviewed').iterdir()) == []
    with app.app_context():
        assert ReviewExample.query.count() == 0
        assert db.session.get(DetectionEvent, ids['passed']).review_status == 'predicted'


def test_legacy_migration_preserves_existing_data(app, client, video_bytes):
    upload_video(client, video_bytes)
    with app.app_context():
        video = Video.query.one()
        video.confidence = .3
        video.sample_seconds = .5
        db.session.commit()
        identifier = video.id
        db.session.remove()
        for name in ('pipeline_version', 'temporal_min_hits', 'temporal_iou', 'temporal_max_gap'):
            db.session.execute(db.text(f'ALTER TABLE video DROP COLUMN {name}'))
        db.session.execute(db.text('ALTER TABLE detection_event DROP COLUMN filter_status'))
        db.session.commit()
        migrate_early_schema()
        migrate_early_schema()
        video = db.session.get(Video, identifier)
        assert video.confidence == .3 and video.pipeline_version == 'legacy'
        assert video.temporal_min_hits == 1 and video.temporal_max_gap == pytest.approx(1.05)
