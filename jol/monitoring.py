import io
import os
import uuid
from flask import Blueprint, abort, flash, redirect, render_template, request, send_file, url_for
from sqlalchemy.exc import IntegrityError

from .models import db, DetectionEvent, RoadSegment, RepairContract, MonitoringReport, active_events
from .routes import login_required
from .transparency import validate_contract, analyze_segment
from .reports import snapshot, report_text, complaint_text, pdf_bytes

bp = Blueprint('monitoring', __name__)


@bp.get('/transparency')
@login_required
def transparency_page():
    segments = RoadSegment.query.order_by(RoadSegment.name).all()
    analyses = [(s, analyze_segment(s, s.events, s.contracts)) for s in segments]
    return render_template('transparency.html', segments=segments,
                           contracts=RepairContract.query.order_by(RepairContract.created_at.desc()).all(),
                           analyses=analyses)


@bp.post('/contracts')
@login_required
def contract_create():
    try:
        values = validate_contract(request.form)
        db.get_or_404(RoadSegment, values['segment_id'])
        db.session.add(RepairContract(**values))
        db.session.commit()
        flash('Шарт жазбасы сақталды.', 'success')
    except (ValueError, IntegrityError) as exc:
        db.session.rollback()
        message = str(exc) if isinstance(exc, ValueError) else 'Бұл нөмірлі шарт бұрын енгізілген.'
        flash(message, 'error')
        return render_template('contract_error.html'), 400
    return redirect(url_for('monitoring.transparency_page'))


@bp.post('/contracts/<int:contract_id>/verify')
@login_required
def contract_verify(contract_id):
    contract = db.get_or_404(RepairContract, contract_id)
    status = request.form.get('status')
    note = request.form.get('note', '').strip()
    if status not in {'verified', 'unverified'} or not note or len(note) > 3000:
        abort(400)
    from .models import now
    contract.verification_status = status
    contract.verification_note = note
    contract.verified_at = now() if status == 'verified' else None
    db.session.commit()
    flash('Шарттың тексеру күйі жаңартылды. Бұрынғы есептер тарихи көшірме ретінде сақталады.', 'success')
    return redirect(url_for('monitoring.transparency_page'))


@bp.route('/reports', methods=['GET', 'POST'])
@login_required
def reports_page():
    if request.method == 'POST':
        raw_id = request.form.get('segment_id')
        try:
            segment = db.get_or_404(RoadSegment, int(raw_id)) if raw_id else None
        except ValueError:
            abort(400)
        events = DetectionEvent.query.filter(active_events()).filter_by(segment_id=segment.id if segment else None).all()
        contracts = segment.contracts if segment else []
        data = snapshot(segment, events, contracts, os.getenv('PUBLIC_BASE_URL', 'http://127.0.0.1:5000'))
        report = MonitoringReport(id=uuid.uuid4().hex, segment_id=segment.id if segment else None,
                                  title=data['segment'], snapshot=data)
        db.session.add(report)
        db.session.commit()
        return redirect(url_for('monitoring.report_detail', report_id=report.id))
    return render_template('reports.html', segments=RoadSegment.query.order_by(RoadSegment.name).all(),
                           reports=MonitoringReport.query.order_by(MonitoringReport.created_at.desc()).all())


@bp.get('/reports/<report_id>')
@login_required
def report_detail(report_id):
    report = db.get_or_404(MonitoringReport, report_id)
    return render_template('report.html', report=report, text=report_text(report.snapshot),
                           complaint=complaint_text(report.snapshot))


@bp.get('/reports/<report_id>/pdf')
@login_required
def export_pdf(report_id):
    report = db.get_or_404(MonitoringReport, report_id)
    response = send_file(io.BytesIO(pdf_bytes(report.snapshot)), mimetype='application/pdf', as_attachment=True,
                         download_name=f'JolBaqylau-{report.id[:8]}.pdf')
    response.headers['Cache-Control'] = 'private, no-store'
    return response


@bp.get('/reports/<report_id>/complaint.txt')
@login_required
def export_complaint(report_id):
    report = db.get_or_404(MonitoringReport, report_id)
    return send_file(io.BytesIO(complaint_text(report.snapshot).encode('utf-8')), mimetype='text/plain; charset=utf-8',
                     as_attachment=True, download_name=f'otinish-{report.id[:8]}.txt')
