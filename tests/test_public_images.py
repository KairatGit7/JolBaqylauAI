import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from jol.models import db, Video, ImageAnalysis


def token(client, route='/images'):
    assert client.get(route).status_code == 200
    with client.session_transaction() as session:
        assert not session.get('admin')
        return session['csrf']


def image_bytes(kind='PNG'):
    data = io.BytesIO()
    Image.new('RGB', (80, 60), '#778899').save(data, kind)
    return data.getvalue()


def test_public_pages_keep_old_data_and_admin_actions_private(app, client, video_bytes):
    from conftest import upload_video
    private = upload_video(client, video_bytes).location
    anonymous = app.test_client()
    for route in ['/', '/analysis', '/images', '/about']:
        page = anonymous.get(route)
        assert page.status_code == 200
        assert b'type="password"' not in page.data
        assert private.encode() not in page.data
    csrf = token(anonymous)
    for route in ['/map', '/reports', '/transparency', '/reviews', '/reviews/export', private]:
        assert anonymous.get(route).status_code == 302
    for route in ['/segments', '/contracts', '/events/1/review']:
        assert anonymous.post(route, data={'csrf_token': csrf}).status_code == 302
    assert anonymous.get('/instance/admin-password.txt').status_code == 404
    assert anonymous.post('/analysis', data={'csrf_token': 'missing'}).status_code == 400


def test_video_without_password_and_session_isolation(app, video_bytes):
    first, second = app.test_client(), app.test_client()
    csrf = token(first, '/analysis')
    response = first.post('/analysis', data={'csrf_token': csrf, 'observed_at': '2026-01-01T12:00',
                                            'video': (io.BytesIO(video_bytes), 'road.avi')})
    assert response.status_code == 302
    assert first.get(response.location).status_code == 200
    assert second.get(response.location).status_code == 302
    identifier = response.location.rsplit('/', 1)[-1]
    assert first.get(f'/api/videos/{identifier}/status').status_code == 200
    assert second.get(f'/api/videos/{identifier}/status').status_code == 401
    assert first.get(f'/raw/{identifier}').data == video_bytes
    assert second.get(f'/raw/{identifier}').status_code == 302
    folder = app.config['STORAGE'] / 'evidence' / identifier
    folder.mkdir()
    (folder / 'frame.jpg').write_bytes(b'owned evidence')
    assert first.get(f'/evidence/{identifier}/frame.jpg').data == b'owned evidence'
    assert second.get(f'/evidence/{identifier}/frame.jpg').status_code == 302
    csrf2 = token(second, '/analysis')
    response2 = second.post('/analysis', data={'csrf_token': csrf2, 'observed_at': '2026-01-01T12:00',
                                             'video': (io.BytesIO(video_bytes), 'road.avi')})
    assert response2.location != response.location  # No checksum-based private record leakage.
    with app.app_context():
        assert Video.query.count() == 2


@pytest.mark.parametrize('suffix,kind', [('jpg', 'JPEG'), ('jpeg', 'JPEG'), ('png', 'PNG'), ('webp', 'WEBP')])
def test_image_formats_annotation_and_access(app, monkeypatch, suffix, kind):
    calls = []
    def predict(frame, confidence):
        calls.append((frame.shape, confidence))
        return [dict(category='pothole', confidence=.75, bbox=[10, 20, 50, 45])]
    monkeypatch.setattr(app.extensions['road_detector'], 'predict', predict)
    client = app.test_client()
    csrf = token(client)
    response = client.post('/images', data={'csrf_token': csrf, 'image': (io.BytesIO(image_bytes(kind)), f'road.{suffix}')})
    assert response.status_code == 302
    result = client.get(response.location)
    assert result.status_code == 200 and 'Ойық' in result.data.decode()
    assert calls == [((60, 80, 3), .6)]
    annotated = client.get(response.location + '/annotated.png?download=1')
    original = client.get(response.location + '/original.png')
    assert annotated.status_code == 200 and annotated.data.startswith(b'\x89PNG')
    assert annotated.headers['Content-Disposition'].startswith('attachment;')
    assert annotated.headers['Cache-Control'] == 'private, no-store'
    assert not np.array_equal(cv2.imdecode(np.frombuffer(original.data, np.uint8), 1),
                              cv2.imdecode(np.frombuffer(annotated.data, np.uint8), 1))
    stranger = app.test_client()
    assert stranger.get(response.location).status_code == 404
    assert stranger.get(response.location + '/annotated.png').status_code == 404
    with app.app_context():
        assert ImageAnalysis.query.one().predictions[0]['confidence'] == .75
        assert Video.query.count() == 0


def test_invalid_images_never_invoke_model(app, monkeypatch):
    def fail(*args):
        pytest.fail('Invalid image must not invoke YOLO')
    monkeypatch.setattr(app.extensions['road_detector'], 'predict', fail)
    client = app.test_client()
    csrf = token(client)
    for name, contents in [('bad.jpg', b'not an image'), ('test.svg', b'<svg/>'),
                           ('mismatch.jpg', image_bytes('PNG')), ('cut.png', image_bytes()[:30])]:
        assert client.post('/images', data={'csrf_token': csrf,
                           'image': (io.BytesIO(contents), name)}).status_code == 400
    app.config['MAX_IMAGE_BYTES'] = 20
    assert client.post('/images', data={'csrf_token': csrf,
                       'image': (io.BytesIO(image_bytes()), 'large.png')}).status_code == 400
    app.config['MAX_IMAGE_BYTES'] = 1024 * 1024
    app.config['MAX_IMAGE_PIXELS'] = 100
    assert client.post('/images', data={'csrf_token': csrf,
                       'image': (io.BytesIO(image_bytes()), 'pixels.png')}).status_code == 400
    with app.app_context():
        assert ImageAnalysis.query.count() == 0
    assert not list((app.config['STORAGE'] / 'images').iterdir())


def test_exif_orientation_and_empty_results(app, monkeypatch):
    calls = []
    def predict(frame, confidence):
        calls.append(frame.shape)
        return []
    monkeypatch.setattr(app.extensions['road_detector'], 'predict', predict)
    image = Image.new('RGB', (80, 60), 'white')
    exif = Image.Exif()
    exif[274] = 6
    data = io.BytesIO()
    image.save(data, 'JPEG', exif=exif)
    client = app.test_client()
    response = client.post('/images', data={'csrf_token': token(client),
                           'image': (io.BytesIO(data.getvalue()), 'oriented.jpg')})
    assert calls == [(80, 60, 3)]
    page = client.get(response.location)
    assert 'жол ақауы анықталмады' in page.data.decode()
    with Image.open(io.BytesIO(client.get(response.location + '/original.png').data)) as output:
        assert output.size == (60, 80) and not output.getexif()


def test_image_inference_failure_does_not_fake_result(app, monkeypatch):
    def fail(*args):
        raise RuntimeError('test failure')
    monkeypatch.setattr(app.extensions['road_detector'], 'predict', fail)
    client = app.test_client()
    response = client.post('/images', data={'csrf_token': token(client),
                           'image': (io.BytesIO(image_bytes()), 'valid.png')})
    assert response.status_code == 503
    with app.app_context():
        assert ImageAnalysis.query.count() == 0


@pytest.mark.integration
def test_real_image_inference_without_login(app):
    # Real cached checkpoint; explicit test setting avoids losing this known ~0.54 example.
    app.config['MODEL_CONFIDENCE'] = .25
    client = app.test_client()
    path = Path(__file__).resolve().parents[1] / 'samples/rdd2022/China_Drone_000101.jpg'
    response = client.post('/images', data={'csrf_token': token(client),
                           'image': (io.BytesIO(path.read_bytes()), path.name)})
    assert response.status_code == 302
    with app.app_context():
        result = ImageAnalysis.query.one()
        assert any(p['category'] == 'pothole' for p in result.predictions)
        assert all(0 <= p['confidence'] <= 1 for p in result.predictions)
    assert 'Ойық' in client.get(response.location).data.decode()
    assert client.get(response.location + '/annotated.png').status_code == 200
