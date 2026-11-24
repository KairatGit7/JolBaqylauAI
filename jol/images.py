"""Single photographs: validate/decode, real inference, session-private PNG results."""
import io
import uuid
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.Image import open as open_image  # Keep native validation; YOLO patches Image.open with optional plugin installation.
from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, send_from_directory, url_for

from .models import db, ImageAnalysis, ImageReviewExample
from .routes import can_access, visitor_key
from .vision import annotate
from .presets import inference_settings

bp = Blueprint('images', __name__)
FORMATS = {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.png': 'PNG', '.webp': 'WEBP'}


def decode_image(upload):
    extension = Path(upload.filename or '').suffix.lower() if upload else ''
    if extension not in FORMATS:
        raise ValueError('JPG, JPEG, PNG немесе WEBP суретін таңдаңыз.')
    contents = upload.stream.read(current_app.config['MAX_IMAGE_BYTES'] + 1)
    if len(contents) > current_app.config['MAX_IMAGE_BYTES']:
        raise ValueError('Сурет көлемі 10 МБ-тан аспауы тиіс.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with open_image(io.BytesIO(contents)) as image:
                if image.format != FORMATS[extension]:
                    raise ValueError('Файл кеңейтімі нақты сурет түріне сәйкес емес.')
                if image.width * image.height > current_app.config['MAX_IMAGE_PIXELS']:
                    raise ValueError('Сурет 16 мегапиксельден аспауы тиіс.')
                if getattr(image, 'is_animated', False):
                    raise ValueError('Анимация емес, бір фотосурет жүктеңіз.')
                image.verify()
            with open_image(io.BytesIO(contents)) as image:
                # Match camera orientation; retain aspect ratio, strip metadata and composite transparency.
                oriented = ImageOps.exif_transpose(image).convert('RGBA')
                background = Image.new('RGBA', oriented.size, 'white')
                rgb = Image.alpha_composite(background, oriented).convert('RGB')
                return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('Сурет ашылмады. Бүтін JPG, PNG немесе WEBP файлын таңдаңыз.') from exc


@bp.route('/images', methods=['GET', 'POST'])
def image_analysis():
    if request.method == 'POST':
        created = []
        try:
            settings = inference_settings()
            upload = request.files.get('image')
            frame = decode_image(upload)
            kwargs = {'imgsz': settings['inference_size']} if settings['inference_size'] != 640 else {}
            predictions = current_app.extensions['road_detector'].predict(frame, settings['confidence'], **kwargs)
            identifier = uuid.uuid4().hex
            for suffix, pixels in [('original', frame), ('annotated', annotate(frame, predictions))]:
                path = current_app.config['STORAGE'] / 'images' / f'{identifier}-{suffix}.png'
                created.append(path)
                if not cv2.imwrite(str(path), pixels):
                    raise OSError('Сурет нәтижесін сақтау мүмкін болмады.')
                path.chmod(0o600)
            result = ImageAnalysis(id=identifier, owner_key=visitor_key(), name=(upload.filename or 'Сурет')[:200],
                                   width=frame.shape[1], height=frame.shape[0], **settings,
                                   model_revision=current_app.config['MODEL_REVISION'], predictions=predictions)
            db.session.add(result)
            db.session.commit()
            return redirect(url_for('images.image_result', image_id=identifier))
        except ValueError as exc:
            flash(str(exc), 'error')
            return render_template('images.html'), 400
        except Exception:
            db.session.rollback()
            for path in created:
                path.unlink(missing_ok=True)
            current_app.logger.exception('Суретті талдау орындалмады')
            flash('Суретті талдау орындалмады. Модельдің қолжетімділігін және сервер журналын тексеріңіз.', 'error')
            return render_template('images.html'), 503
    return render_template('images.html')


def accessible_image(image_id):
    result = db.session.get(ImageAnalysis, image_id)
    if not result or not can_access(result):
        abort(404)
    return result


@bp.get('/images/<image_id>')
def image_result(image_id):
    result = accessible_image(image_id)
    reviews = {}
    for review in ImageReviewExample.query.filter_by(image_id=image_id).order_by(ImageReviewExample.created_at.desc(), ImageReviewExample.id.desc()).all():
        reviews.setdefault(review.prediction_index, review)
    return render_template('image_result.html', result=result, reviews=reviews)


@bp.post('/images/<image_id>/predictions/<int:prediction_index>/review')
def review_image_prediction(image_id, prediction_index):
    from sqlalchemy.exc import SQLAlchemyError
    from .reviews import archive_image_review
    result = accessible_image(image_id)
    if not 0 <= prediction_index < len(result.predictions):
        abort(404)
    verdict = request.form.get('verdict')
    note = request.form.get('note', '').strip()
    if verdict not in {'confirmed', 'rejected', 'predicted'} or len(note) > 2000:
        abort(400)
    if verdict != 'predicted' and not note:
        flash('Тексеру негізін жазыңыз.', 'error')
        return redirect(url_for('images.image_result', image_id=image_id))
    path = None
    try:
        example, path = archive_image_review(result, prediction_index, verdict, note, request.form.get('reason', 'other'))
        db.session.add(example)
        db.session.commit()
        flash('Тексеру нәтижесі және бөлек бағалау үлгісі сақталды.', 'success')
    except (ValueError, OSError, SQLAlchemyError) as exc:
        db.session.rollback()
        if path:
            path.unlink(missing_ok=True)
        flash(f'Тексеру сақталмады: {exc}', 'error')
    return redirect(url_for('images.image_result', image_id=image_id) + f'#prediction-{prediction_index}')


@bp.get('/images/<image_id>/<variant>.png')
def image_file(image_id, variant):
    accessible_image(image_id)
    if variant not in {'original', 'annotated'}:
        abort(404)
    download = variant == 'annotated' and request.args.get('download') == '1'
    return send_from_directory(current_app.config['STORAGE'] / 'images', f'{image_id}-{variant}.png',
                               mimetype='image/png', as_attachment=download,
                               download_name='JolBaqylau-result.png' if download else None)
