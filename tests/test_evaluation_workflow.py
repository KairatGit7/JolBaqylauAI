import io
import json
from pathlib import Path
import zipfile

import pytest
from PIL import Image

from jol.models import db, ImageAnalysis, GroundTruthSample, DetectionEvaluation
from test_public_images import token

BOX = {'category': 'pothole', 'bbox': [10, 20, 50, 45]}


def photo(client, csrf, color='red'):
    content = io.BytesIO()
    Image.new('RGB', (80, 60), color).save(content, 'PNG')
    return client.post('/images', data={'csrf_token': csrf, 'image': (io.BytesIO(content.getvalue()), 'fixture.png')}).location


def sample_for(client, csrf, image_url):
    return client.post('/evaluation/images/' + image_url.rsplit('/', 1)[-1], data={'csrf_token': csrf}).location


def save(app, client, csrf, url, boxes=None, negative=False, road='fixture-road', split='val', **changes):
    with app.app_context():
        revision = db.session.get(GroundTruthSample, url.rsplit('/', 1)[-1]).revision
    data = dict(csrf_token=csrf, boxes=json.dumps([BOX] if boxes is None else boxes), negative='yes' if negative else '',
                complete='yes', road_group=road, split=split, revision=str(revision))
    data.update(changes)
    return client.post(url, data=data)


@pytest.fixture
def setup(app, monkeypatch):
    monkeypatch.setattr(app.extensions['road_detector'], 'predict', lambda *args, **kwargs: [])
    client = app.test_client()
    return client, token(client)


def test_gt_validation_negative_and_separation(app, setup):
    client, csrf = setup
    image_url = photo(client, csrf)
    url = sample_for(client, csrf, image_url)
    for change in [dict(complete=''), dict(boxes='[]'), dict(negative='yes'), dict(road_group=''),
                   dict(boxes='not json'), dict(boxes=json.dumps([dict(BOX, bbox=[-1, 0, 10, 10])])),
                   dict(boxes=json.dumps([dict(BOX, bbox=[True, 0, 10, 10])])), dict(revision='bad')]:
        assert save(app, client, csrf, url, **change).status_code == 400
    assert save(app, client, csrf, url).status_code == 302
    with app.app_context():
        sample = GroundTruthSample.query.one()
        assert sample.boxes == [BOX] and sample.revision == 1
        assert ImageAnalysis.query.one().predictions == []  # A missed AI defect was labelled independently.
    assert save(app, client, csrf, url, revision='0').status_code == 400
    assert save(app, client, csrf, url, boxes=[], negative=True).status_code == 302
    with app.app_context():
        assert GroundTruthSample.query.one().negative is True
        assert GroundTruthSample.query.one().boxes == []
    assert client.get(url).status_code == 200


def test_yolo_export_positive_negative_and_private_scope(app, setup):
    client, csrf = setup
    positive = sample_for(client, csrf, photo(client, csrf, 'red'))
    negative = sample_for(client, csrf, photo(client, csrf, 'blue'))
    save(app, client, csrf, positive, road='positive')
    save(app, client, csrf, negative, boxes=[], negative=True, road='negative', split='test')
    response = client.get('/evaluation/export')
    assert response.status_code == 200 and response.headers['Cache-Control'] == 'private, no-store'
    archive = zipfile.ZipFile(io.BytesIO(response.data))
    manifest = json.loads(archive.read('manifest.json'))
    pos = next(row for row in manifest if not row['negative'])
    neg = next(row for row in manifest if row['negative'])
    values = archive.read(f'labels/val/{pos["id"]}.txt').decode().split()
    assert values[0] == '3' and len(values) == 5
    assert list(map(float, values[1:])) == pytest.approx([.375, 32.5/60, .5, 25/60])
    assert archive.read(f'labels/test/{neg["id"]}.txt') == b''
    assert archive.read(f'images/val/{pos["id"]}.png').startswith(b'\x89PNG')
    assert b'pothole' in archive.read('data.yaml')
    stranger = app.test_client()
    assert stranger.get(positive).status_code == 404
    assert stranger.get(positive + '/image.png').status_code == 404
    assert stranger.get('/evaluation/export').status_code == 302
    assert pos['id'] not in stranger.get('/evaluation').data.decode()
    assert client.post(positive, data={'boxes': '[]'}).status_code == 400


def test_source_and_road_groups_never_cross_splits(app, setup, video_bytes):
    client, csrf = setup
    response = client.post('/analysis', data={'csrf_token': csrf, 'observed_at': '2026-01-01T12:00',
                                             'video': (io.BytesIO(video_bytes), 'fixture.avi')})
    video_id = response.location.rsplit('/', 1)[-1]
    frames = [client.post('/evaluation/videos/' + video_id, data={'csrf_token': csrf, 'frame_index': str(index)}).location
              for index in (0, 1)]
    save(app, client, csrf, frames[0], road='Shared Road', split='val')
    # New draft frames inherit the source split and are never exported as negatives.
    draft = client.post('/evaluation/videos/' + video_id, data={'csrf_token': csrf, 'frame_index': '2'}).location
    with app.app_context():
        assert db.session.get(GroundTruthSample, draft.rsplit('/', 1)[-1]).split == 'val'
    assert client.get('/evaluation/export').status_code == 200
    save(app, client, csrf, frames[1], road='Other road', split='val')
    linked = sample_for(client, csrf, photo(client, csrf, 'red'))
    save(app, client, csrf, linked, road='  SHARED   ROAD  ', split='test')
    with app.app_context():
        assert {sample.split for sample in GroundTruthSample.query.all()} == {'test'}
    archive = zipfile.ZipFile(io.BytesIO(client.get('/evaluation/export').data))
    manifest = json.loads(archive.read('manifest.json'))
    assert len(manifest) == 3
    assert len({row['group'] for row in manifest}) == 1 and {row['split'] for row in manifest} == {'test'}
    # Identical image across repeated model analyses reuses the same truth, not another split.
    assert sample_for(client, csrf, photo(client, csrf, 'red')) == linked
    assert client.post('/evaluation/videos/' + video_id, data={'csrf_token': csrf, 'frame_index': '-1'}).status_code == 302


def test_compare_same_validation_data_one_to_one_metrics_and_snapshot(app, setup, monkeypatch):
    client, csrf = setup
    positive = sample_for(client, csrf, photo(client, csrf, 'red'))
    negative = sample_for(client, csrf, photo(client, csrf, 'blue'))
    save(app, client, csrf, positive, road='positive')
    save(app, client, csrf, negative, boxes=[], negative=True, road='negative')
    calls = []
    def predict(frame, confidence, *, imgsz):
        is_positive = bool(frame[0, 0, 2] > 100)
        calls.append((is_positive, confidence, imgsz))
        detection = dict(BOX, confidence=.8)
        if confidence == .15:
            return [detection, dict(detection, confidence=.7)] if is_positive else [detection]
        return [detection] if is_positive else []
    monkeypatch.setattr(app.extensions['road_detector'], 'predict', predict)
    response = client.post('/evaluation/compare', data={'csrf_token': csrf})
    assert response.status_code == 302 and '/reports/' in response.location
    snapshot = client.get(response.location + '.json').json
    assert len(calls) == 4
    for is_positive in (True, False):
        assert (is_positive, .15, 640) in calls and (is_positive, .4, 960) in calls
    high, balanced = snapshot['metrics']['high_recall'], snapshot['metrics']['balanced']
    assert (high['true_positive'], high['false_positive'], high['false_negative']) == (1, 2, 0)
    assert high['precision'] == pytest.approx(1/3) and high['recall'] == 1 and high['f1'] == .5
    assert balanced['precision'] == balanced['recall'] == balanced['f1'] == 1
    assert snapshot['scope'] == 'frame_detection_without_temporal_filter'
    assert client.get(response.location).status_code == 200
    assert 'Валидация есебі' in client.get(response.location).data.decode()
    save(app, client, csrf, positive, boxes=[dict(BOX, bbox=[0, 0, 5, 5])], road='positive')
    assert client.get(response.location + '.json').json == snapshot
    assert app.test_client().get(response.location).status_code == 404


def test_no_unlabelled_fake_metrics_or_partial_failed_report(app, setup, monkeypatch):
    client, csrf = setup
    url = sample_for(client, csrf, photo(client, csrf))
    assert client.post('/evaluation/compare', data={'csrf_token': csrf}).status_code == 302
    with app.app_context():
        assert DetectionEvaluation.query.count() == 0
    save(app, client, csrf, url)
    def fail(*args, **kwargs):
        raise RuntimeError('test model failure')
    monkeypatch.setattr(app.extensions['road_detector'], 'predict', fail)
    assert client.post('/evaluation/compare', data={'csrf_token': csrf}).status_code == 503
    with app.app_context():
        assert DetectionEvaluation.query.count() == 0
        sample = GroundTruthSample.query.one()
        (app.config['STORAGE'] / 'ground_truth' / f'{sample.id}.png').write_bytes(b'tampered')
    assert client.post('/evaluation/compare', data={'csrf_token': csrf}).status_code == 302
    assert client.get('/evaluation/export').status_code == 302


def test_negative_only_metrics_undefined_not_perfect(app, setup):
    client, csrf = setup
    url = sample_for(client, csrf, photo(client, csrf))
    save(app, client, csrf, url, boxes=[], negative=True)
    response = client.post('/evaluation/compare', data={'csrf_token': csrf})
    assert response.status_code == 302
    for result in client.get(response.location + '.json').json['metrics'].values():
        assert result['precision'] is None and result['recall'] is None and result['f1'] is None
    assert 'Есептелмейді' in client.get(response.location).data.decode()


@pytest.mark.integration
def test_real_cached_presets_compare_on_labeled_validation(app):
    from scripts.evaluate import read_truth
    import cv2
    client = app.test_client()
    csrf = token(client)
    path = Path(__file__).resolve().parents[1] / 'samples/rdd2022/China_Drone_000101.jpg'
    image_url = client.post('/images', data={'csrf_token': csrf,
        'image': (io.BytesIO(path.read_bytes()), path.name)}).location
    url = sample_for(client, csrf, image_url)
    frame = cv2.imread(str(path))
    truth = read_truth(path.with_suffix('.txt'), frame.shape[1], frame.shape[0])
    assert save(app, client, csrf, url, boxes=truth, road='Published RDD2022 test fixture').status_code == 302
    response = client.post('/evaluation/compare', data={'csrf_token': csrf})
    assert '/reports/' in response.location
    snapshot = client.get(response.location + '.json').json
    assert snapshot['dataset'][0]['truth'] == truth
    assert set(snapshot['predictions']) == {'high_recall', 'balanced'}
    assert all(result['true_positive'] >= 1 for result in snapshot['metrics'].values())
