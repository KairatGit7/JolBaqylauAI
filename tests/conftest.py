import io
import cv2
import numpy as np
import pytest

from jol import create_app


@pytest.fixture
def app(tmp_path):
    return create_app({'TESTING': True, 'SECRET_KEY': 'test-only', 'ADMIN_PASSWORD': 'test-password',
                       'SQLALCHEMY_DATABASE_URI': f'sqlite:///{tmp_path}/test.db',
                       'STORAGE': tmp_path / 'private', 'START_WORKER': False})


@pytest.fixture
def client(app):
    client = app.test_client()
    with client.session_transaction() as session:
        session['admin'] = True
        session['csrf'] = 'test-csrf'
    return client


@pytest.fixture
def video_bytes(tmp_path):
    path = tmp_path / 'test.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (160, 120))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((120, 160, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def upload_video(client, contents, name='test.avi'):
    return client.post('/analysis', data={'csrf_token': 'test-csrf',
        'observed_at': '2026-01-01T12:00', 'video': (io.BytesIO(contents), name)})
