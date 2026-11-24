"""Exercise review buttons against an isolated database; never label production data."""
import io
import json
from pathlib import Path
import shutil
import tempfile
import threading
import zipfile
from datetime import datetime

from playwright.sync_api import sync_playwright, expect
from werkzeug.serving import make_server

from jol import create_app
from jol.models import db, Video, ProcessingJob, DetectionEvent, ReviewExample

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='review-browser-', dir='/tmp/opencode') as temporary:
        storage = Path(temporary)
        app = create_app({'TESTING': True, 'START_WORKER': False, 'STORAGE': storage,
                          'SQLALCHEMY_DATABASE_URI': f'sqlite:///{storage}/test.db',
                          'SECRET_KEY': 'isolated-browser-only', 'ADMIN_PASSWORD': 'isolated-browser-only'})
        shutil.copyfile(ROOT / 'samples/DEMO_RDD2022_slideshow.avi', storage / 'raw/demo.avi')
        # Known demo prediction; these are UI test fixtures, never evaluation labels.
        with app.app_context():
            video = Video(id='browser-fixture', name='UI TEST ONLY', stored_name='demo.avi',
                          observed_at=datetime(2026, 1, 1), duration=12, fps=10, width=640, height=680,
                          sample_seconds=1, confidence=.6, model_revision=app.config['MODEL_REVISION'], is_demo=True)
            db.session.add(video)
            db.session.add(ProcessingJob(video=video, status='completed', progress=100, frames_analyzed=12))
            event = DetectionEvent(video=video, category='pothole', confidence=.6782191395759583,
                       start_seconds=9, end_seconds=9, bbox=[36.42, 48.65, 253.39, 213.42],
                       hits=[dict(timestamp=9, confidence=.6782191395759583, bbox=[36.42, 48.65, 253.39, 213.42])],
                       evidence_name='fixture.jpg', filter_status='suppressed')
            db.session.add(event)
            db.session.commit()
            event_id = event.id
        import cv2
        cap = cv2.VideoCapture(str(storage / 'raw/demo.avi'))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 90)
        ok, frame = cap.read()
        cap.release()
        assert ok and cv2.imwrite(str(storage / 'evidence/fixture.jpg'), frame)
        server = make_server('127.0.0.1', 0, app, threaded=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        errors = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                context = browser.new_context(viewport={'width': 1440, 'height': 1000}, locale='kk-KZ')
                page = context.new_page()
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(base + '/login')
                page.locator('#password').fill('isolated-browser-only')
                page.get_by_role('button', name='Жүйеге кіру').click()
                page.wait_for_url(base + '/')
                page.goto(base + '/videos/browser-fixture')
                form = page.locator('.review-form')
                form.locator('[name=note]').fill('UI TEST ONLY: exercise rejected verdict, not a ground-truth label.')
                form.locator('[name=reason]').select_option('manhole_cover')
                form.get_by_role('button', name='Қате анықталды').click()
                expect(page.locator('.event-card .badge')).to_contain_text('Қабылданбаған')
                assert context.request.get(base + '/api/events').json() == []
                with app.app_context():
                    example = ReviewExample.query.one()
                    assert example.verdict == 'rejected' and example.reason == 'manhole_cover'
                    snapshot_id = example.id
                page.goto(base + '/reviews')
                expect(page.get_by_role('heading', name='Тексерілген үлгілер')).to_be_visible()
                expect(page.locator('img')).to_have_attribute('src', f'/reviews/{snapshot_id}/image')
                archive = zipfile.ZipFile(io.BytesIO(context.request.get(base + '/reviews/export').body()))
                record = json.loads(archive.read('reviews.jsonl'))
                assert record['verdict'] == 'rejected' and record['exhaustively_labeled'] is False
                assert archive.read(record['image']).startswith(b'\x89PNG')
                for width in (320, 390, 768):
                    page.set_viewport_size({'width': width, 'height': 844})
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
                page.goto(base + '/videos/browser-fixture')
                form = page.locator('.review-form')
                form.locator('[name=note]').fill('UI TEST ONLY: exercise human rescue.')
                form.get_by_role('button', name='Дұрыс анықталды').click()
                expect(page.locator('.event-card .badge')).to_contain_text('Расталған')
                assert len(context.request.get(base + '/api/events?status=confirmed').json()) == 1
                with app.app_context():
                    assert ReviewExample.query.count() == 2
                    assert db.session.get(DetectionEvent, event_id).filter_status == 'suppressed'
                    assert db.session.get(ReviewExample, snapshot_id).verdict == 'rejected'
                form = page.locator('.review-form')
                form.locator('details > summary').click()
                form.locator('[name=status]').select_option('predicted')
                form.get_by_role('button', name='Тексеруді сақтау').click()
                expect(page.locator('.event-card .badge')).to_contain_text('Модель болжамы')
                archive = zipfile.ZipFile(io.BytesIO(context.request.get(base + '/reviews/export').body()))
                assert archive.read('reviews.jsonl') == b''
                anonymous = browser.new_context()
                for path in ('/reviews', '/reviews/export', f'/reviews/{snapshot_id}/image'):
                    assert anonymous.request.get(base + path, max_redirects=0).status == 302
                browser.close()
            assert not errors, errors
            result = dict(status='passed', isolated_database=True, production_labels_written=0, js_errors=errors,
                          checks=['named submit buttons with JavaScript enabled', 'immutable review archive',
                                  'rejected reason and raw PNG export', 'suppressed candidate rescue',
                                  'retraction removes exported label', 'private access', '320/390/768px review layout'])
            (ROOT / 'artifacts/detection-review-browser.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2))
        finally:
            server.shutdown()
            thread.join(timeout=5)


if __name__ == '__main__':
    main()
