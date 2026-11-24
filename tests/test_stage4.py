from pathlib import Path
import cv2
import pytest
from jol.evaluation import evaluate, match_boxes
from jol.models import Video
from jol.transparency import analyze_segment
from conftest import upload_video


def box(category='pothole', confidence=.8, coords=None):
    return {'category': category, 'confidence': confidence, 'bbox': coords or [0, 0, 10, 10]}


def test_detection_metrics():
    records = [{'predictions': [box(), box(confidence=.7), box('transverse_crack')],
                'truth': [box(), box(coords=[100, 100, 110, 110])]}]
    result = evaluate(records)
    assert result['true_positive'] == 1
    assert result['false_positive'] == 2
    assert result['false_negative'] == 1
    assert result['precision'] == pytest.approx(1/3)
    assert result['recall'] == .5
    assert result['f1'] == .4
    assert match_boxes([box()], [box('transverse_crack')]) == (0, 1, 1)
    assert evaluate([])['precision'] is None


def test_duplicate_upload(client, app, video_bytes):
    first = upload_video(client, video_bytes)
    second = upload_video(client, video_bytes)
    assert first.location == second.location
    with app.app_context():
        assert Video.query.count() == 1


def test_demo_excluded_from_conclusions():
    from test_stage3 import fake_event, contract_data
    from types import SimpleNamespace as NS
    from jol.transparency import validate_contract
    event = fake_event()
    event.video.is_demo = True
    findings = analyze_segment(NS(name='Test'), [event], [NS(id=1, **validate_contract(contract_data()))])
    assert {f['code'] for f in findings} == {'DEMO'}


def test_private_headers_and_missing_records(client):
    assert client.get('/videos/not-found').status_code == 404
    assert client.get('/reports/not-found/pdf').status_code == 404
    response = client.get('/')
    assert response.headers['X-Frame-Options'] == 'DENY'
    assert 'no-store' in response.headers['Cache-Control']


def test_vibration_not_pothole_classification(client):
    from jol.vibration import analyze_vibration
    contents = 'timestamp,acceleration_x,acceleration_y,acceleration_z\n'
    contents += '\n'.join(f'{i},0,0,{30 if i == 10 else 9.8}' for i in range(20))
    result = analyze_vibration(contents.encode())
    assert len(result['spikes']) == 1
    assert result['spikes'][0]['latitude'] is None
    assert 'category' not in result['spikes'][0]
    assert client.get('/vibration').status_code == 200


@pytest.mark.integration
def test_real_road_sample(app):
    path = Path(__file__).resolve().parents[1] / 'samples/rdd2022/China_Drone_000101.jpg'
    assert path.exists(), 'scripts.fetch_assets арқылы лицензияланған суреттерді жүктеңіз.'
    frame = cv2.imread(str(path))
    predictions = app.extensions['road_detector'].predict(frame, .3)
    assert any(p['category'] == 'pothole' for p in predictions)
    assert all(0 <= p['confidence'] <= 1 for p in predictions)
