from datetime import datetime
import io
import pytest
from jol.gps import coordinates, parse_track, synchronize
from jol.models import db, Video, DetectionEvent, RoadSegment, GPSObservation
from conftest import upload_video


def test_gps_synchronization():
    points = [(0, 45., 78.), (2, 45.2, 78.4)]
    assert synchronize(points, 0, offset=1) == pytest.approx((45.1, 78.2))
    assert synchronize(points, 2) == (45.2, 78.4)
    assert synchronize(points, 3) is None
    assert synchronize([], 0) is None
    assert synchronize([(0, 0, 0), (10, 1, 1)], 5) is None


def test_parse_csv_and_gpx():
    origin = datetime(2026, 1, 1, 7)
    points = parse_track(b'timestamp,latitude,longitude\n2026-01-01T12:00:00+05:00,45,78', '.csv', origin)
    assert points == [(0, 45, 78)]
    gpx = b'<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg><trkpt lat="45" lon="78"><time>2026-01-01T07:00:00Z</time></trkpt></trkseg></trk></gpx>'
    assert parse_track(gpx, '.gpx', origin) == points
    with pytest.raises(ValueError):
        coordinates('nan', 12)
    with pytest.raises(ValueError):
        parse_track(b'timestamp,latitude,longitude\n0,45,78\n0,45,78', '.csv', origin)
    with pytest.raises(ValueError):
        parse_track(b'<!DOCTYPE gpx><gpx/>', '.gpx', origin)


@pytest.fixture
def event(app, client, video_bytes):
    response = upload_video(client, video_bytes)
    video_id = response.location.rsplit('/', 1)[1]
    with app.app_context():
        video = db.session.get(Video, video_id)
        video.job.status = 'completed'
        video.is_current = True
        item = DetectionEvent(video_id=video_id, category='pothole', confidence=.8,
            start_seconds=0, end_seconds=1, bbox=[0, 0, 50, 50], evidence_name='test.jpg',
            hits=[{'timestamp': 0, 'confidence': .8, 'bbox': [0,0,50,50], 'video_id': video_id}])
        db.session.add(item)
        db.session.commit()
        return item.id, video_id


def test_unlocated_and_manual_segment(client, app, event):
    event_id, video_id = event
    assert client.get('/map').status_code == 200
    data = client.get('/api/events').json[0]
    assert data['latitude'] is None
    assert client.post('/segments', data={'csrf_token': 'test-csrf', 'name': 'Сынақ учаскесі'}).status_code == 302
    with app.app_context():
        segment_id = RoadSegment.query.one().id
    assert client.post(f'/events/{event_id}/location', data={'csrf_token': 'test-csrf', 'segment_id': segment_id}).status_code == 302
    assert client.get('/api/events').json[0]['latitude'] is None
    assert client.get(f'/videos/{video_id}').status_code == 200
    invalid = client.post(f'/events/{event_id}/location', data={'csrf_token': 'test-csrf', 'latitude': '45', 'longitude': '78'})
    assert invalid.status_code == 400


def test_import_gps(client, app, event):
    event_id, video_id = event
    response = client.post(f'/videos/{video_id}/gps', data={'csrf_token': 'test-csrf', 'verified': 'yes',
        'offset': '0', 'source': 'ТЕК СИНТЕТИКАЛЫҚ ТЕСТ',
        'gps': (io.BytesIO(b'timestamp,latitude,longitude\n0,45,78\n2,45.2,78.4'), 'gps.csv')})
    assert response.status_code == 302
    assert client.get('/api/events').json[0]['latitude'] == 45
    with app.app_context():
        assert GPSObservation.query.count() == 2
    assert client.get('/api/events?category=bad').status_code == 400
    assert client.get('/api/events?status=confirmed').json == []
