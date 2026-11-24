from datetime import date, datetime
from types import SimpleNamespace as NS
from decimal import Decimal
import pytest

from jol.models import db, RoadSegment, RepairContract, MonitoringReport
from jol.transparency import validate_contract, analyze_segment, official_url
from jol.reports import snapshot, report_text, pdf_bytes, complaint_text


def contract_data(**updates):
    data = dict(identifier='SYNTHETIC-TEST-ONLY', title='СИНТЕТИКАЛЫҚ ТЕСТ', value='100.00',
                customer='Сынақ ұйымы', contractor='Сынақ мердігері', segment_id='1',
                start_date='2025-01-01', completion_date='2025-06-01', warranty_end='2027-06-01',
                warranty_note='Синтетикалық тест шарты', source_url='https://satypalu.gov.kz/kz/registry/contract',
                verification_status='verified', verified='yes', verification_note='ТЕК АВТОМАТТЫ ТЕСТ')
    data.update(updates)
    return data


def test_contract_validation():
    assert validate_contract(contract_data())['value'] == Decimal('100')
    for update in [{'value': '-1'}, {'value': 'NaN'}, {'value': '1.001'}, {'source_url': 'javascript:alert(1)'},
                   {'source_url': 'https://satypalu.gov.kz.evil.test/x'}, {'verified': ''},
                   {'completion_date': '2024-01-01'}, {'warranty_note': ''}, {'segment_id': 'abc'}]:
        with pytest.raises(ValueError):
            validate_contract(contract_data(**update))
    assert not official_url('https://user:password@gov.kz/x')


def fake_event(id=1, status='confirmed', day=1):
    return NS(id=id, video_id=f'v{id}', review_status=status, start_seconds=0,
              video=NS(observed_at=datetime(2026, 1, day)))


def test_contract_comparison():
    c = NS(id=1, **validate_contract(contract_data()))
    events = [fake_event(), fake_event(2, day=2), fake_event(3, 'predicted')]
    findings = analyze_segment(NS(name='Тест'), events, [c])
    assert {f['code'] for f in findings} == {'A', 'B', 'C', 'P'}
    assert all(3 not in f['event_ids'] for f in findings if f['code'] in {'A', 'B', 'C'})
    findings = analyze_segment(NS(name='Тест'), [fake_event(status='rejected')], [c])
    assert not any(f['code'] in {'A', 'B', 'C'} for f in findings)
    assert {f['code'] for f in analyze_segment(None, [], [])} == {'D'}
    c.verification_status = 'unverified'
    assert not any(f['code'] in {'A', 'C'} for f in analyze_segment(NS(name='Тест'), events, [c]))


def test_same_day_not_independent():
    findings = analyze_segment(NS(name='Тест'), [fake_event(1), fake_event(2)], [])
    assert 'B' not in {f['code'] for f in findings}


def test_contract_database_and_reports(client, app):
    client.post('/segments', data={'csrf_token': 'test-csrf', 'name': 'СИНТЕТИКАЛЫҚ ТЕСТ'})
    data = dict(contract_data(), csrf_token='test-csrf')
    assert client.post('/contracts', data=data).status_code == 302
    assert client.post('/contracts', data=data).status_code == 400
    assert client.post('/contracts', data=dict(data, value='invalid')).status_code == 400
    assert client.get('/transparency').status_code == 200
    response = client.post('/reports', data={'csrf_token': 'test-csrf', 'segment_id': '1'})
    assert response.status_code == 302
    assert client.get(response.location).status_code == 200
    pdf = client.get(response.location + '/pdf')
    assert pdf.status_code == 200 and pdf.data.startswith(b'%PDF-')
    complaint = client.get(response.location + '/complaint.txt').data.decode('utf-8')
    assert 'автоматты жіберу орындалмайды' in complaint
    with app.app_context():
        report = MonitoringReport.query.one()
        assert report.snapshot['contracts'][0]['identifier'] == 'SYNTHETIC-TEST-ONLY'
        contract = RepairContract.query.one()
        contract.title = 'Өзгерген атау'
        db.session.commit()
        assert report.snapshot['contracts'][0]['title'] == 'СИНТЕТИКАЛЫҚ ТЕСТ'


def test_empty_report():
    data = snapshot(None, [], [], 'http://127.0.0.1:5000')
    assert 'Анықталмаған' not in data['segment']
    assert 'сыбайлас жемқорлықтың дәлелі емес' in report_text(data)
    assert len(pdf_bytes(data)) > 1000
