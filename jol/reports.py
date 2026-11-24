import io
from datetime import timedelta
from xml.sax.saxutils import escape, quoteattr

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

from .models import now
from .transparency import analyze_segment
from .vision import LABELS, font_path

LIMITATIONS = ('Бұл есеп — мониторинг материалы. Жол ақауы сыбайлас жемқорлықтың дәлелі емес. '
    'Болжам сенімділігі ақиқаттың ықтималдығы емес. Кадрлық біріктіру шамамен орындалады. '
    'GPS камераның орнын береді; координатсыз жазба картаға қойылмайды. '
    'Жөндеудің сапасын, ақаудың себебін, тереңдігін және қаржылық шығынды осы деректермен дәл анықтау мүмкін емес. '
    'Шарт деректері қолмен тексеріледі; API қосылмаған. Түсірілім күні мен учаскені әкімші береді. '
    'Қазақстан жолдарындағы дәлдік жеткілікті тәуелсіз таңбаланған жиынтықпен расталмаған.')
STATUS_LABELS = {'predicted': 'Модель болжамы', 'confirmed': 'Адам растаған бақылау', 'rejected': 'Қабылданбаған болжам'}


def snapshot(segment, events, contracts, base_url):
    return {
        'version': 1, 'created_at': now().isoformat() + 'Z',
        'segment': segment.name if segment else 'Учаскесі тағайындалмаған оқиғалар',
        'events': [dict(id=e.id, video_id=e.video_id, video_name=e.video.name, is_demo=e.video.is_demo, category=LABELS[e.category],
                        confidence=e.confidence, review_status=STATUS_LABELS[e.review_status], review_note=e.review_note,
                        observed_at=(e.video.observed_at + timedelta(seconds=e.start_seconds, hours=5)).isoformat() + '+05:00',
                        seconds=e.start_seconds, end_seconds=e.end_seconds, hits=len(e.hits), bbox=e.bbox,
                        latitude=e.latitude, longitude=e.longitude, location_source=e.location_source,
                        evidence=base_url.rstrip('/') + '/evidence/' + e.evidence_name,
                        detail=base_url.rstrip('/') + '/videos/' + e.video_id + f'#event-{e.id}',
                        model_revision=e.video.model_revision, pipeline_version=e.video.pipeline_version,
                        confidence_threshold=e.video.confidence, temporal_min_hits=e.video.temporal_min_hits,
                        filter_status=e.filter_status) for e in events],
        'contracts': [dict(id=c.id, identifier=c.identifier, title=c.title, value=str(c.value),
                           customer=c.customer, contractor=c.contractor,
                           start=str(c.start_date or 'Белгісіз'), completion=str(c.completion_date or 'Белгісіз'),
                           warranty_end=str(c.warranty_end or 'Белгісіз'), warranty_note=c.warranty_note,
                           source_url=c.source_url, verification_status=c.verification_status,
                           verification_note=c.verification_note) for c in contracts],
        'findings': analyze_segment(segment, events, contracts), 'limitations': LIMITATIONS,
        'explanation_method': 'AI болжамдарына негізделген ережелік түсіндірме; генеративті тілдік модель қолданылмайды.',
    }


def report_text(data):
    lines = ['ЖОЛБАҚЫЛАУ AI — МОНИТОРИНГ ЕСЕБІ', 'Учаске: ' + data['segment'],
             'Есеп жасалған уақыт: ' + data['created_at'],
             'Түсіндірме әдісі: ' + data['explanation_method'], '', 'БАҚЫЛАУ ДӘЛЕЛДЕРІ']
    if not data['events']:
        lines.append('Осы учаскеге бақылау енгізілмеген.')
    for e in data['events']:
        location = f"{e['latitude']:.6f}, {e['longitude']:.6f} ({e['location_source']})" if e['latitude'] is not None else 'Анықталмаған'
        lines.extend([f"#{e['id']} · {e['category']} · {e['review_status']}",
            'Дерек түрі: ' + ('ДЕМОНСТРАЦИЯ — түсірілім күні шартты; нақты жол мониторингі емес' if e.get('is_demo') else 'Пайдаланушы ұсынған бақылау'),
            f"Түсірілім: {e['observed_at']}; бейне: {e['video_name']} (ID: {e['video_id']})",
            f"Уақыт: {e['seconds']:.2f}–{e['end_seconds']:.2f} с; біріктірілген белгілеу: {e['hits']}",
            f"Ең жоғары сенімділік: {e['confidence']:.1%}; қорап: {e['bbox']}",
            'Орналасу: ' + location, 'Тексеру негізі: ' + (e['review_note'] or 'Тексерілмеген'),
            'Дәлел: ' + e['evidence'], 'Жазба: ' + e['detail'], 'Модель нұсқасы: ' + e['model_revision'], ''])
    lines.append('ЖӨНДЕУ ШАРТТАРЫ')
    if not data['contracts']:
        lines.append('Жүйеге енгізілген шарт жоқ. Бұл шарт мүлде жоқ деген қорытынды емес.')
    for c in data['contracts']:
        verified = 'Әкімші ресми дерекпен тексерген' if c['verification_status'] == 'verified' else 'Тексерілмеген'
        lines.extend([f"{c['identifier']} · {c['title']} · {c['value']} ₸", 'Күйі: ' + verified,
                      f"Тапсырыс беруші: {c['customer']}; мердігер: {c['contractor']}",
                      f"Мерзім: {c['start']} — {c['completion']}; кепілдік: {c['warranty_end']}",
                      'Кепілдік негізі: ' + c['warranty_note'], 'Тексеру негізі: ' + c['verification_note'],
                      'Ресми дереккөз: ' + c['source_url'], ''])
    lines.append('ТЕКСЕРУ ҰСЫНЫМДАРЫ')
    for f in data['findings']:
        lines.extend([f"[{f['code']}] {f['title']}", f['text'],
                      'Дәлел оқиғалары: ' + (', '.join(map(str, f['event_ids'])) or 'Қолданылмайды')])
        if f['source_url']:
            lines.append('Шарт көзі: ' + f['source_url'])
    if not data['findings']:
        lines.append('Берілген деректерде ережелік тексеру шарты іске қосылмады. Бұл сапа кепілдігі емес.')
    lines.extend(['', 'ШЕКТЕУЛЕР', data['limitations'], '',
                  'Дәлел сілтемелері жергілікті қорғалған жүйеге апарады. Сыртқа жіберерде дербес мәліметті бүркемелеңіз.'])
    return '\n'.join(lines)


def complaint_text(data):
    return ('ӨТІНІШ ЖОБАСЫ — пайдаланушы тексеріп, өзі жібереді\n\n'
        'Кімге: [құзыретті органды толтырыңыз]\nКімнен: [аты-жөніңіз және байланыс деректеріңіз]\n\n'
        f"Тақырыбы: {data['segment']} учаскесінің жай-күйін тексеру туралы\n\n"
        'Көрсетілген жол учаскесіндегі тіркелген бақылауларды қарауды, жөндеу жұмысының '
        'көлемі мен қабылдау құжаттары туралы қолжетімді ақпарат беруді және қажет болған жағдайда '
        'жол қауіпсіздігі бойынша шара қабылдауды сұраймын.\n'
        'Қосымша: төмендегі мониторинг есебі. Модель болжамдары мен адам растаған жазбалар бөлек көрсетілген. '
        'Бұл өтініште қандай да бір тұлғаның құқық бұзғаны туралы тұжырым жасалмайды.\n\n'
        + report_text(data) + '\n\nКүні: [толтырыңыз]\nҚолы: [толтырыңыз]\n'
        'Жіберу: https://eotinish.kz/ — автоматты жіберу орындалмайды.')


def pdf_bytes(data):
    path = font_path()
    if not path:
        raise RuntimeError('Қазақша PDF үшін DejaVuSans.ttf қарпі қажет.')
    if 'DejaVuKZ' not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont('DejaVuKZ', path))
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, title='ЖолБақылау AI — мониторинг есебі',
                                 author='ЖолБақылау AI', leftMargin=40, rightMargin=40,
                                 topMargin=38, bottomMargin=38)
    style = ParagraphStyle('KZ', fontName='DejaVuKZ', fontSize=9, leading=14,
                           textColor=colors.HexColor('#1b344a'), alignment=TA_LEFT, spaceAfter=6)
    story = []
    for line in report_text(data).splitlines():
        markup = escape(line)
        for prefix in ('Дәлел: ', 'Жазба: ', 'Ресми дереккөз: ', 'Шарт көзі: '):
            if line.startswith(prefix):
                url = line[len(prefix):]
                if url.startswith(('http://', 'https://')):
                    markup = escape(prefix) + f'<link href={quoteattr(url)} color="#147e96">{escape(url)}</link>'
        story.append(Paragraph(markup, style) if line else Spacer(1, 8))
    document.build(story)
    return buffer.getvalue()
