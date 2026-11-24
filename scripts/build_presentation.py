"""Нақты артефакттардан 12 слайдтық қазақша PPTX жасау."""
import json
import io
from pathlib import Path
from xml.etree import ElementTree

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
NAVY = '#101F34'
WHITE = '#F4F8FC'
CYAN = '#65DDD4'
MUTED = '#A7BED0'


def color(value):
    return RGBColor.from_string(value.lstrip('#'))


def text(slide, value, x, y, w, h, size=20, fill=WHITE, bold=False):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    frame.word_wrap = True
    for index, line in enumerate(value.split('\n')):
        p = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        p.text = line
        p.font.name = 'DejaVu Sans'
        p.font.size = Pt(size)
        p.font.color.rgb = color(fill)
        p.font.bold = bold
        p.space_after = Pt(14)
    return shape


def rectangle(slide, x, y, w, h, fill):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color(fill)
    shape.line.fill.background()
    return shape


def image(slide, relative, x, y, w, h):
    from PIL import Image
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(f'Нақты визуал жоқ: {path}; алдымен browser_smoke орындаңыз.')
    bitmap = Image.open(path)
    if 'screenshots/' in relative:
        # Тек нақты скриншоттың үзіндісі; мазмұн қосылмайды және өзгертілмейді.
        crops = {'map.png': (290, 295, 1410, 1035),
                 'dashboard.png': (270, 95, 1420, 850),
                 'reports.png': (270, 100, 1420, 890),
                 'upload.png': (270, 100, 1420, 1000)}
        if path.name in crops:
            bitmap = bitmap.crop(crops[path.name])
    width, height = bitmap.size
    stream = io.BytesIO()
    bitmap.save(stream, format='PNG')
    stream.seek(0)
    scale = min(w / width, h / height)
    slide.shapes.add_picture(stream, Inches(x + (w-width*scale)/2), Inches(y+(h-height*scale)/2),
                             width=Inches(width*scale), height=Inches(height*scale))


def main():
    evaluation = json.loads((ROOT / 'artifacts/evaluation-rdd2022-4.json').read_text())
    junit = ElementTree.parse(ROOT / 'artifacts/pytest-results.xml').getroot()
    suites = list(junit.iter('testsuite'))
    passed = sum(int(s.get('tests', 0))-int(s.get('failures', 0))-int(s.get('errors', 0))-int(s.get('skipped', 0)) for s in suites)
    browser = json.loads((ROOT / 'artifacts/browser-check.json').read_text())
    if browser['status'] != 'passed':
        raise ValueError('Браузерлік тексеру сәтті аяқталмаған.')
    slides = [
        ('ЖолБақылау AI', 'Жол сапасы мен мемлекеттік қаражаттың\nашықтығын бақылау жүйесі',
         ['Бейне → дәлел → ашық дерек → жауапты шешім', 'Қазақстан · 2026 · Студенттік AI жобасы'], 'artifacts/screenshots/dashboard.png'),
        ('Мәселе: деректер арасындағы алшақтық', 'Неге бұл жоба қажет?',
         ['Әр ақауды азаматтың жеке түсіруіне тәуелді болмау.', 'Жолдағы бақылау мен жөндеу құжатын байланыстыру.', 'Қоғамдық тексеруге уақыт, орын және көрнекі дәлел беру.'], None),
        ('Ашықтық — алдын алудың негізі', 'Жол ақауы сыбайлас жемқорлықты дәлелдемейді',
         ['Міндеттеме мен нақты бақылау бір дәлел тізбегінде.', 'Дереккөз, белгісіздік және тексеру негізі ашық.', 'Мақсат: тәуелсіз тексеруге негізделген сұрақ.', 'Кінә, жемқорлық ықтималдығы, қаржылық шығын есептелмейді.'], None),
        ('Нақты жасанды интеллект', 'RDD2022 жол ақаулары моделі · YOLOv8n',
         ['Бойлық / көлденең / торлы жарықшақ / ойық.', 'best.pt · бекітілген модель нұсқасы.', 'CPU inference. Қайта үйрету жүргізілген жоқ.', 'Confidence — модель ұпайы, факт кепілі емес.'], 'artifacts/model-check.jpg'),
        ('Бейне → дәлел → оқиға', 'Қарапайым, жергілікті архитектура',
         ['OpenCV: 180 с дейін, кадрларды ағынмен оқу.', 'Әдепкіде әр секундтан бір кадр → YOLO.', 'Уақыт + IoU: қайталанған кадрларды шамамен біріктіру.', 'Фондық кезек → SQLite → JPEG / H.264 шолу.'], None),
        ('Картада тек негізі бар орын', 'GPS болмаса — координатсыз жазба',
         ['CSV / GPX + құжатталған уақыт ығысуы.', 'Трек сыртында / >5 с үзілісте координат жоқ.', 'GPS — камера орны, ақаудың дәл нүктесі емес.', 'Талдықорған — тек бастапқы карта орталығы.'], 'artifacts/screenshots/map.png'),
        ('Ашықтық мониторингі', 'Ресми шарттар қолмен тексеріліп енгізіледі',
         ['A · Жөндеуден кейінгі расталған бақылау.', 'B · Әртүрлі сапарлар мен күндердегі қайталану.', 'C · Құжатталған кепілдік кезіндегі бақылау.', 'D · Жетіспейтін дерек. API / генеративті LLM қосылмаған.'], None),
        ('Адам шешімі. Тексерілетін есеп.', 'Болжам мен расталған бақылау бөлек',
         ['Қазақша PDF: дәлел, орын, шарт, ұсыным.', 'Есеп — жасалған сәттегі өзгермейтін көшірме.', 'Демо бақылаулар A / B / C талдауына кірмейді.', 'eOtinish өтінішін пайдаланушы өзі жібереді.'], 'artifacts/screenshots/reports.png'),
        ('Бағалау: нақты әрі шектеулі', '4 RDD2022 test суреті · Қазақстаннан емес',
         [f"TP={evaluation['true_positive']} · FP={evaluation['false_positive']} · FN={evaluation['false_negative']} · GT=6",
          f"Precision {evaluation['precision']:.4f} · Recall {evaluation['recall']:.4f} · F1 {evaluation['f1']:.4f}",
          'IoU ≥ 0.5 · confidence ≥ 0.30 · бір-біріне сәйкестік.',
          'Жергілікті дәлдік әлі валидацияланбаған. Бұл mAP емес.'], None),
        ('Тексерілген жұмыс құбыры', 'Тест нәтижелері нақты журналдардан алынды',
         [f'pytest: {passed} тест өтті, оның ішінде нақты модель интеграциясы.',
          'Chromium: жүктеу → AI → карта → қазақша PDF.',
          '390 px мобильді көрініс; жеке бейнеге кіру қорғалған.',
          'Әкімші, CSRF, файл валидациясы; қолмен бүркемелеу көшірмесі.'], None),
        ('Тікелей демонстрация', '12 секунд · лицензияланған RDD2022 слайд-бейне',
         ['Нақты модельдің қорабы және кадрлық дәлелі.', 'Қайталанған кадр ≠ тәуелсіз жаңа ақау.', 'GPS жоқ → картада ойдан нүкте жоқ.', 'PDF және қолмен жіберілетін өтініш жобасы.'], 'artifacts/screenshots/upload.png'),
        ('Сапалы жол. Ашық болашақ.', 'AI + тексерілетін дәлел + ашық міндеттеме',
         ['Қоғамдық бақылаудың қайталанымдылығын қолдау.', 'Жергілікті таңбаланған дерек және мамандармен пилот.',
          'Біз айып тақпаймыз. Тексерілетін сұрақ қалыптастырамыз.', 'Назарларыңызға рақмет!'], None),
    ]
    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = Inches(13.333), Inches(7.5)
    for index, (title, subtitle, bullets, picture) in enumerate(slides, 1):
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = color(NAVY)
        rectangle(slide, .55, .48, .14, .32, CYAN)
        text(slide, 'ЖОЛБАҚЫЛАУ AI   /   2026', .82, .4, 10, .4, 11, CYAN, True)
        text(slide, title, .65, 1.08, 12, 1, 31, WHITE, True)
        text(slide, subtitle, .68, 2.0, 12, .9 if index == 1 else .65, 17, MUTED)
        body_y = 3.15 if index == 1 else 2.9
        if picture:
            rectangle(slide, 6.85, 2.95, 5.85, 3.75, '#1B334A')
            image(slide, picture, 6.98, 3.08, 5.58, 3.43)
            text(slide, '\n'.join('•  ' + b for b in bullets), .72, body_y, 5.65, 3.55, 19)
            caption = 'Нақты модель нәтижесі · RDD2022 · CC BY-SA 4.0' if index == 4 else 'Нақты интерфейс скриншотының үзіндісі · демо'
            text(slide, caption, 6.97, 6.72, 5.7, .3, 8, MUTED)
        else:
            for j, bullet in enumerate(bullets):
                rectangle(slide, .72, body_y+j*.83, .48, .45, '#1D4855')
                text(slide, f'{j+1:02}', .79, body_y+j*.83+.03, .4, .35, 11, CYAN, True)
                text(slide, bullet, 1.42, body_y+j*.83-.01, 10.9, .72, 21)
        text(slide, 'Жол ақауы — сыбайлас жемқорлықтың дәлелі емес.', .7, 7.08, 11, .25, 9, MUTED)
        text(slide, f'{index:02} / 12', 11.85, 7.06, 1, .3, 10, CYAN)
        slide.notes_slide.notes_text_frame.text = subtitle + '\n' + '\n'.join(bullets) + '\nҚорғау мәтіні: DEFENSE_SPEECH_KZ.md. '
    destination = ROOT / 'artifacts/JolBaqylau_AI_2026_KZ.pptx'
    presentation.save(destination)
    print(f'{destination} · {len(presentation.slides)} слайд')


if __name__ == '__main__':
    main()
