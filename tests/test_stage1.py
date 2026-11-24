import io
import numpy as np
import pytest

from jol.models import db, Video, DetectionEvent
from jol.vision import LABELS, category_name, sampled_frames, EventAggregator, inspect_video
from conftest import upload_video


def test_startup(client):
    assert client.get('/health').json['status'] == 'ok'
    for page in ['/', '/analysis', '/about']:
        assert client.get(page).status_code == 200


def test_auth_and_csrf(app, client):
    anonymous = app.test_client()
    assert anonymous.get('/raw/private').status_code == 302
    assert anonymous.get('/evidence/private.jpg').status_code == 302
    assert client.post('/analysis', data={}).status_code == 400
    assert anonymous.get('/api/videos/x/status').status_code == 401


def test_upload_validation(client, video_bytes, app):
    assert upload_video(client, b'not video', 'attack.mp4').status_code == 400
    assert upload_video(client, video_bytes, 'video.exe').status_code == 400
    response = upload_video(client, video_bytes, '../../test.avi')
    assert response.status_code == 302
    with app.app_context():
        video = Video.query.one()
        assert '/' not in video.stored_name
        assert video.duration == pytest.approx(2)
        assert video.observed_at.hour == 7
        assert video.job.status == 'queued'
    assert client.get(response.location).status_code == 200


def test_size_limit(app, client):
    app.config['MAX_CONTENT_LENGTH'] = 200
    assert upload_video(client, b'x' * 201).status_code == 413


def test_frame_extraction(tmp_path, video_bytes):
    path = tmp_path / 'video.avi'
    path.write_bytes(video_bytes)
    frames = list(sampled_frames(path, 0.5))
    assert [t for _, t, _ in frames] == [0, .5, 1, 1.5]
    assert all(frame.shape == (120, 160, 3) for _, _, frame in frames)
    with pytest.raises(ValueError):
        inspect_video(path, max_seconds=1)


def test_labels():
    assert category_name('D40') == 'pothole'
    assert category_name('Longitudinal Crack') == 'longitudinal_crack'
    assert len(LABELS) == 4
    with pytest.raises(ValueError):
        category_name('car')


def test_aggregation():
    aggregator = EventAggregator()
    d = {'category': 'pothole', 'bbox': [0, 0, 50, 50], 'confidence': .8}
    aggregator.add([d, dict(d, bbox=[80, 80, 100, 100])], 0, 'a.jpg', 'v')
    aggregator.add([dict(d, bbox=[1, 1, 51, 51])], 1, 'b.jpg', 'v')
    assert len(aggregator.events) == 2
    assert len(aggregator.events[0]['hits']) == 2
    aggregator.add([d], 10, 'c.jpg', 'v')
    assert len(aggregator.events) == 3


def test_worker_failure(app, client, video_bytes, monkeypatch):
    response = upload_video(client, video_bytes)
    def fail():
        raise RuntimeError('Модель жүктелмеді: offline')
    monkeypatch.setattr(app.extensions['road_detector'], 'load', fail)
    video_id = response.location.rsplit('/', 1)[1]
    app.extensions['job_runner'].process(video_id)
    result = client.get(f'/api/videos/{video_id}/status').json
    assert result['status'] == 'failed'
    assert 'offline' in result['error']
    with app.app_context():
        assert DetectionEvent.query.count() == 0


@pytest.mark.integration
def test_actual_model_load_and_inference(app):
    detector = app.extensions['road_detector']
    model = detector.load()
    assert {category_name(n) for n in model.names.values()} == set(LABELS)
    predictions = detector.predict(np.zeros((640, 640, 3), dtype=np.uint8))
    assert isinstance(predictions, list)
    for item in predictions:
        assert 0 <= item['confidence'] <= 1
        assert len(item['bbox']) == 4


@pytest.mark.integration
def test_real_worker(app, client, video_bytes):
    response = upload_video(client, video_bytes)
    video_id = response.location.rsplit('/', 1)[1]
    app.extensions['job_runner'].process(video_id)
    status = client.get(f'/api/videos/{video_id}/status').json
    assert status['status'] == 'completed', status
    assert status['frames'] == 2
    assert client.get(response.location).status_code == 200
