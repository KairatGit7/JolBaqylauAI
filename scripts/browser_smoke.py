"""Verify the real local app and save current-design screenshots (no invented records)."""
import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, expect
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
BASE = os.getenv('BROWSER_BASE_URL', 'http://127.0.0.1:5000')
OUT = ROOT / 'artifacts/redesign'
SHOTS = OUT / 'screenshots'
SHOTS.mkdir(parents=True, exist_ok=True)
AXE = ROOT / '.bootstrap/axe.min.js'
errors, accessibility, checks = [], {}, []


def audit(page, name):
    if not AXE.exists():
        return
    result = page.evaluate("""async () => {
      const result = await axe.run(document, {runOnly: {type: 'tag',
        values: ['wcag2a', 'wcag2aa', 'wcag21aa', 'best-practice']}});
      return {violations: result.violations, incomplete: result.incomplete,
              passes: result.passes.length, version: axe.version};
    }""")
    accessibility[name] = result
    (OUT / 'accessibility.json').write_text(json.dumps(accessibility, ensure_ascii=False, indent=2))


def screenshot(page, name):
    page.evaluate('document.fonts.ready')
    page.screenshot(path=SHOTS / f'{name}.png', full_page=True)


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(viewport={'width': 1440, 'height': 1050}, locale='kk-KZ')
    if AXE.exists():
        context.add_init_script(path=AXE)
    page = context.new_page()
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(BASE + '/login')
    page.get_by_role('button', name='Жүйеге кіру').click()
    expect(page.locator('#password')).to_be_focused()
    expect(page.locator('#password-error')).to_have_text('Бұл өрісті толтырыңыз.')
    audit(page, 'login-validation')
    screenshot(page, 'login')
    page.locator('[name=password]').fill((ROOT / 'instance/admin-password.txt').read_text().strip())
    page.get_by_role('button', name='Жүйеге кіру').click()
    page.wait_for_url(BASE + '/')
    checks.append('Kazakh inline validation, first-error focus, authenticated login')

    page.goto(BASE + '/')
    page.keyboard.press('Tab')
    expect(page.locator('.skip')).to_be_focused()
    page.keyboard.press('Enter')
    expect(page.locator('#main')).to_be_focused()
    checks.append('Keyboard skip-to-content link')

    page.goto(BASE + '/analysis')
    page.get_by_role('button', name='Талдауды бастау').click()
    expect(page.locator('#video-file')).to_be_focused()
    page.locator('#video-file').set_input_files({'name': 'invalid.txt', 'mimeType': 'text/plain', 'buffer': b'not a video'})
    expect(page.locator('#video-file-error')).to_contain_text('MP4, MOV немесе AVI')
    page.locator('#video-file').set_input_files(ROOT / 'samples/DEMO_RDD2022_slideshow.avi')
    expect(page.locator('#file-info')).to_contain_text('DEMO_RDD2022_slideshow.avi')
    page.locator('#upload-zone').evaluate("""zone => {
      const data = new DataTransfer();
      data.items.add(document.querySelector('#video-file').files[0]);
      zone.dispatchEvent(new DragEvent('drop', {dataTransfer: data, bubbles: true, cancelable: true}));
    }""")
    page.locator('[name=observed_at]').fill('2026-01-01T12:00')
    page.locator('[name=is_demo]').check()
    dialogs = []
    def dismiss_unsaved(dialog):
        dialogs.append(dialog.type)
        dialog.dismiss()
    page.once('dialog', dismiss_unsaved)
    page.locator('.sidebar nav a[href="/about"]').click()
    assert dialogs == ['beforeunload']
    assert '/analysis' in page.url
    page.get_by_role('button', name='Талдауды бастау').click()
    page.wait_for_url('**/videos/**')
    video_url = page.url
    reused = page.locator('.flash-message').filter(has_text='бұрын жүктелген').count() > 0
    expect(page.locator('#job-status')).to_have_attribute('data-state', re.compile('completed|failed'), timeout=180000)
    assert page.locator('#job-status').get_attribute('data-state') == 'completed'
    assert page.locator('.event-card').count() > 0
    video_id = urlparse(video_url).path.rsplit('/', 1)[1]
    status = context.request.get(BASE + f'/api/videos/{video_id}/status').json()
    events = context.request.get(BASE + '/api/events').json()
    demo_events = [event for event in events if f'/videos/{video_id}#' in event['detail']]
    assert demo_events and all(event['is_demo'] for event in demo_events)
    assert all(event['latitude'] is None and event['longitude'] is None for event in demo_events)
    evidence = context.request.get(BASE + demo_events[0]['evidence'])
    assert evidence.ok
    assert all(event['width'] > 0 and event['height'] > 0 for event in demo_events)
    anonymous = browser.new_context()
    assert anonymous.request.get(BASE + f'/raw/{video_id}', max_redirects=0).status == 302
    assert anonymous.request.get(BASE + demo_events[0]['evidence'], max_redirects=0).status == 302
    assert anonymous.request.get(BASE + '/api/events').status == 401
    anonymous.close()
    checks.append('Upload validation, drag/drop, unsaved-input guard, real demo result, protected evidence')

    first = page.locator('.event-card').first
    panel = first.locator('details[data-url-panel]').first
    panel.locator('summary').click()
    panel_id = panel.get_attribute('id')
    expect(page).to_have_url(re.compile(r'panel=' + panel_id))
    page.reload()
    expect(page.locator('#' + panel_id)).to_have_attribute('open', '')
    audit(page, 'video-evidence-expanded')
    page.locator('.event-card').first.screenshot(path=SHOTS / 'evidence-card.png')
    page.goto(video_url)
    page.screenshot(path=SHOTS / 'analysis-result.png', full_page=False)
    checks.append('Evidence panel deep links survive reload')

    # A missing review justification must fail without changing the stored prediction.
    predicted = next((event for event in demo_events if event['review_status'] == 'predicted'), None)
    if predicted:
        form = page.locator(f'#event-{predicted["id"]} .review-form')
        form.locator('[name=note]').fill('')
        form.get_by_role('button', name='Дұрыс анықталды').click()
        expect(page.locator('.flash-message.error')).to_contain_text('Тексеру негізін жазыңыз')
        expect(page.locator('.flash-message.error')).to_be_focused()
        current = context.request.get(BASE + '/api/events').json()
        assert next(event for event in current if event['id'] == predicted['id'])['review_status'] == 'predicted'
        checks.append('Unjustified confirmation rejected; original observation preserved')

    routes = [('/', 'dashboard'), ('/analysis', 'upload'), ('/map', 'map'),
              ('/transparency', 'transparency'), ('/reports', 'reports'),
              ('/about', 'about'), ('/vibration', 'vibration')]
    for route, name in routes:
        assert page.goto(BASE + route).status == 200
        if route == '/map':
            expect(page.locator('#map-status')).to_contain_text('Сүзгі нәтижесі')
            assert page.locator('#map-records tr').count() == len(events)
            confirmed = [event for event in events if event['review_status'] == 'confirmed']
            page.locator('#status-filter').select_option('confirmed')
            expect(page).to_have_url(re.compile('status=confirmed'))
            expect(page.locator('#map-status')).to_contain_text(f'{len(confirmed)} жазба')
            page.reload()
            expect(page.locator('#status-filter')).to_have_value('confirmed')
            category = events[0]['category']
            page.locator('#category-filter').select_option(category)
            expect(page).to_have_url(re.compile('category=' + category))
            page.locator('#reset-filters').click()
            expect(page).to_have_url(BASE + '/map')
            expect(page.locator('#map-records tr')).to_have_count(len(events))
            page.go_back()
            expect(page.locator('#status-filter')).to_have_value('confirmed')
            expect(page.locator('#category-filter')).to_have_value(category)
            page.locator('#reset-filters').click()
            expect(page.locator('#map-records tr')).to_have_count(len(events))
            # Exercise actual network failure and user-driven recovery.
            page.route('**/api/events?*', lambda route: route.abort())
            page.locator('#status-filter').select_option('rejected')
            expect(page.locator('#map-records')).to_have_attribute('aria-busy', 'false')
            expect(page.locator('#map-status')).not_to_contain_text('Сүзгі нәтижесі')
            page.unroute('**/api/events?*')
            page.locator('#reset-filters').click()
            expect(page.locator('#map-records tr')).to_have_count(len(events))
            checks.append('Map filters, URL reload/back/reset, empty state, request-failure recovery')
        audit(page, name)
        screenshot(page, name)

    page.goto(BASE + '/reports')
    page.get_by_role('button', name='Есепті қалыптастыру').click()
    page.wait_for_url(re.compile(BASE + r'/reports/[a-f0-9]+$'))
    report_url = page.url
    pdf = context.request.get(report_url + '/pdf')
    assert pdf.status == 200 and pdf.body().startswith(b'%PDF-')
    text = '\n'.join(item.extract_text() for item in PdfReader(io.BytesIO(pdf.body())).pages)
    assert 'ЖОЛБАҚЫЛАУ AI' in text and 'ДЕМОНСТРАЦИЯ' in text
    (OUT / 'demo-monitoring-report.pdf').write_bytes(pdf.body())
    page.locator('#complaint > summary').click()
    context.grant_permissions(['clipboard-read', 'clipboard-write'])
    page.locator('[data-copy]').click()
    expect(page.locator('#announcements')).to_have_text('Өтініш мәтіні көшірілді.')
    assert page.evaluate('navigator.clipboard.readText()') == page.locator('#complaint-text').text_content()
    complaint = context.request.get(report_url + '/complaint.txt')
    assert complaint.ok and complaint.body().decode('utf-8') == page.locator('#complaint-text').text_content()
    audit(page, 'report-detail')
    screenshot(page, 'report-detail')
    checks.append('Report generation, Kazakh PDF text, complaint download and clipboard')

    widths = [320, 390, 768]
    for width in widths:
        page.set_viewport_size({'width': width, 'height': 844})
        for route, name in routes + [(urlparse(video_url).path, 'video'), (urlparse(report_url).path, 'report')]:
            page.goto(BASE + route)
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1'), (width, route)
            if width == 390:
                audit(page, 'mobile-' + name)
                if name in {'dashboard', 'upload', 'map'}:
                    screenshot(page, 'mobile-' + name)
    checks.append('320px, 390px and 768px layouts without page overflow')

    page.set_viewport_size({'width': 390, 'height': 844})
    page.goto(BASE + '/')
    menu = page.locator('#menu-toggle')
    menu.click()
    expect(page.locator('#navigation-dialog')).to_be_visible()
    expect(menu).to_have_attribute('aria-expanded', 'true')
    for _ in range(12):
        page.keyboard.press('Tab')
        assert page.evaluate('document.querySelector("#navigation-dialog").contains(document.activeElement)')
    audit(page, 'mobile-menu')
    page.screenshot(path=SHOTS / 'mobile-menu.png')
    page.keyboard.press('Escape')
    expect(page.locator('#navigation-dialog')).not_to_be_visible()
    expect(menu).to_be_focused()
    menu.click()
    page.get_by_role('navigation', name='Мобильді мәзір').get_by_role('link', name='AI талдау').click()
    expect(page).to_have_url(BASE + '/analysis')
    page.emulate_media(reduced_motion='reduce')
    assert page.evaluate('getComputedStyle(document.documentElement).scrollBehavior') == 'auto'
    checks.append('Mobile menu focus containment, Escape/return focus, navigation, reduced motion')

    # Keep complete automated findings, including checks needing human review.
    (OUT / 'accessibility.json').write_text(json.dumps(accessibility, ensure_ascii=False, indent=2))
    violations = {name: [item['id'] for item in audit_result['violations']]
                  for name, audit_result in accessibility.items() if audit_result['violations']}
    result = {'status': 'passed' if not errors and not violations else 'failed',
              'checked_at': datetime.now(timezone.utc).isoformat(), 'browser': 'Chromium',
              'video_id': video_id, 'job': status, 'reused_existing_analysis': reused,
              'demo_events': len(demo_events), 'demo_unlocated': True,
              'pdf_kazakh_text_verified': True, 'widths': widths, 'js_errors': errors,
              'accessibility_pages': len(accessibility), 'accessibility_violations': violations,
              'checks': checks, 'report_url': report_url,
              'note': 'Real RDD2022 demo evidence. Duplicate upload reuses existing inference. No GPS or contracts invented.'}
    (OUT / 'browser-check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    browser.close()
    assert not errors, errors
    assert not violations, violations
