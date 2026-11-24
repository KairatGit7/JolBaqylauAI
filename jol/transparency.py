from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from urllib.parse import urlparse

from .models import now


def official_url(value):
    try:
        parsed = urlparse(value)
        host = (parsed.hostname or '').lower()
        return (parsed.scheme == 'https' and not parsed.username and not parsed.password
                and parsed.port in (None, 443) and len(value) <= 1000
                and any(host == d or host.endswith('.' + d) for d in ['satypalu.gov.kz', 'goszakup.gov.kz', 'gov.kz']))
    except ValueError:
        return False


def validate_contract(form):
    data = {}
    for key, maximum in [('identifier', 120), ('title', 300), ('customer', 300), ('contractor', 300)]:
        value = form.get(key, '').strip()
        if not value or len(value) > maximum:
            raise ValueError('Шарт нөмірі, атауы, тапсырыс беруші және мердігер міндетті; мәтін тым ұзын болмауы тиіс.')
        data[key] = value
    try:
        amount = Decimal(form.get('value', ''))
        if not amount.is_finite() or amount <= 0 or amount >= Decimal('1000000000000000') or amount.as_tuple().exponent < -2:
            raise ValueError('Шарт сомасы оң сан, ең көбі екі ондық таңбамен беріледі.')
        data['value'] = amount
        data['segment_id'] = int(form.get('segment_id', ''))
        for key in ('start_date', 'completion_date', 'warranty_end'):
            data[key] = date.fromisoformat(form[key]) if form.get(key) else None
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError('Сома, учаске немесе күн форматы жарамсыз.') from exc
    if data['start_date'] and data['completion_date'] and data['completion_date'] < data['start_date']:
        raise ValueError('Аяқталу күні басталу күнінен ерте болмауы тиіс.')
    data['warranty_note'] = form.get('warranty_note', '').strip()
    if len(data['warranty_note']) > 3000:
        raise ValueError('Кепілдік түсіндірмесі тым ұзын.')
    if data['warranty_end'] and (not data['completion_date'] or data['warranty_end'] < data['completion_date'] or not data['warranty_note']):
        raise ValueError('Кепілдік үшін аяқталу күні және ресми құжаттағы негізі қажет.')
    data['source_url'] = form.get('source_url', '').strip()
    if not official_url(data['source_url']):
        raise ValueError('Ресми gov.kz / goszakup.gov.kz / satypalu.gov.kz HTTPS сілтемесін көрсетіңіз.')
    data['verification_status'] = form.get('verification_status', 'unverified')
    if data['verification_status'] not in {'verified', 'unverified'}:
        raise ValueError('Тексеру күйі жарамсыз.')
    data['verification_note'] = form.get('verification_note', '').strip()
    if len(data['verification_note']) > 3000:
        raise ValueError('Тексеру жазбасы тым ұзын.')
    if data['verification_status'] == 'verified':
        if form.get('verified') != 'yes' or not data['verification_note']:
            raise ValueError('Ресми дерекпен салыстыруды растаңыз және құжат деректемесін жазыңыз.')
        data['verified_at'] = now()
    return data


def observation_date(event):
    return (event.video.observed_at + timedelta(seconds=event.start_seconds, hours=5)).date()


def analyze_segment(segment, events, contracts):
    """YOLO бақылауларына негізделген, қайта тексерілетін ережелік түсіндірмелер."""
    findings = []
    events = [e for e in events if getattr(e.video, 'is_current', True)
              and (getattr(e, 'filter_status', 'legacy') != 'suppressed' or e.review_status == 'confirmed')]
    real_events = [e for e in events if not getattr(e.video, 'is_demo', False)]
    confirmed = [e for e in real_events if e.review_status == 'confirmed']
    predicted = [e for e in real_events if e.review_status == 'predicted']
    verified = [c for c in contracts if c.verification_status == 'verified']

    def add(code, kind, title, text, observations=(), contract=None):
        findings.append(dict(code=code, kind=kind, title=title, text=text,
                             event_ids=[e.id for e in observations],
                             contract_id=contract.id if contract else None,
                             source_url=contract.source_url if contract else None))

    if not segment:
        add('D', 'missing_information', 'Учаске тағайындалмаған',
            'Оқиғаны жөндеу шартымен салыстыру үшін учаскені тәуелсіз дерекпен анықтау қажет.')
    if len(real_events) != len(events):
        add('DEMO', 'missing_information', 'Демонстрациялық материал бөлек көрсетілді',
            'Демо бейнелердің шартты күндері мен бақылаулары A/B/C салыстыру ережелеріне кірмейді.')
    if not verified:
        add('D', 'missing_information', 'Тексерілген шарт дерегі жоқ',
            'Жүйеде расталған шарт жоқ. Бұл жол жөнделмеді немесе заң бұзылды дегенді білдірмейді.')
    if predicted:
        add('P', 'model_prediction', 'Модель болжамдары тексеруді күтеді',
            f'{len(predicted)} оқиға адаммен расталмаған. Олар расталған бақылаулар қатарына кірмейді.', predicted)
    # Бір бейнедегі бірнеше кадр немесе бір күндегі қайталау B шартына жеткіліксіз.
    if len({e.video_id for e in confirmed}) >= 2 and len({observation_date(e) for e in confirmed}) >= 2:
        add('B', 'review_recommendation', 'Әртүрлі сапарларда қайталанған бақылау',
            'Кемінде екі бейнеде және екі күнде адам растаған бақылау бар. Бір ақаудың қайталануы ма, '
            'жаңа зақым ба — тәуелсіз тексеру қажет. Бұл жеке ақаулардың дәл саны емес.', confirmed)
    for contract in verified:
        if not contract.start_date or not contract.completion_date:
            add('D', 'missing_information', 'Жөндеу мерзімі толық емес',
                'Жөндеудің нақты орындалған мерзімін ресми қабылдау құжатымен толықтырыңыз.', contract=contract)
        if not contract.warranty_end:
            add('D', 'missing_information', 'Кепілдік мерзімі белгісіз',
                'Жарияланған кепілдік аяқталу күні енгізілмеген; кепілдік бұзылды деп қорытынды жасауға болмайды.', contract=contract)
        if contract.completion_date:
            after = [e for e in confirmed if observation_date(e) > contract.completion_date]
            if after:
                add('A', 'potential_inconsistency', 'Жөндеу аяқталғаннан кейінгі бақылау',
                    'Адам растаған ақау шарттағы аяқталу күнінен кейін тіркелген. Бұл оның жаңадан пайда '
                    'болғанын дәлелдемейді. Жұмыс көлемін, учаске сәйкестігін және қабылдау актісін тексеру ұсынылады.', after, contract)
            warranty = [e for e in confirmed if contract.warranty_end and
                        contract.completion_date <= observation_date(e) <= contract.warranty_end]
            if warranty:
                add('C', 'potential_inconsistency', 'Құжатталған кепілдік кезіндегі бақылау',
                    'Бақылау күні жарияланған кепілдік мерзіміне сәйкес келеді. Ақаудың кепілдікпен қамтылуын '
                    'және себептерін мамандар тексеруі қажет.', warranty, contract)
    return findings
