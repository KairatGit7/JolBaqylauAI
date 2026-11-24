import os
import secrets
from pathlib import Path

from flask import Flask, jsonify, render_template
from sqlalchemy import event
from sqlalchemy.engine import Engine
from werkzeug.security import generate_password_hash

from .models import db, ProcessingJob, migrate_early_schema
from .vision import RoadDetector, LABELS, REVISION


@event.listens_for(Engine, 'connect')
def sqlite_options(connection, _):
    if connection.__class__.__module__.startswith('sqlite3'):
        cursor = connection.cursor()
        cursor.execute('PRAGMA foreign_keys=ON')
        cursor.execute('PRAGMA busy_timeout=10000')
        cursor.close()


def create_app(config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY=os.getenv('SECRET_KEY'),
        SQLALCHEMY_DATABASE_URI='sqlite:///' + str(Path(app.instance_path) / 'jol.db'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        MAX_CONTENT_LENGTH=int(os.getenv('MAX_UPLOAD_MB', '200')) * 1024 * 1024,
        MAX_VIDEO_SECONDS=float(os.getenv('MAX_VIDEO_SECONDS', '180')),
        MAX_IMAGE_BYTES=10 * 1024 * 1024,
        MAX_IMAGE_PIXELS=16_000_000,
        SAMPLE_SECONDS=float(os.getenv('SAMPLE_SECONDS', '1')),
        MODEL_CONFIDENCE=float(os.getenv('MODEL_CONFIDENCE', '0.60')),
        TEMPORAL_MIN_HITS=int(os.getenv('TEMPORAL_MIN_HITS', '2')),
        TEMPORAL_IOU=float(os.getenv('TEMPORAL_IOU', '0.20')),
        MODEL_DEVICE=os.getenv('MODEL_DEVICE', 'cpu'),
        MODEL_REPO=os.getenv('MODEL_REPO', 'dronefreak/rdd2022-yolov8n'),
        MODEL_FILENAME=os.getenv('MODEL_FILENAME', 'best.pt'),
        MODEL_REVISION=os.getenv('MODEL_REVISION', REVISION),
        STORAGE=Path(app.instance_path),
        SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax',
        SESSION_COOKIE_SECURE=os.getenv('COOKIE_SECURE', 'false').lower() == 'true',
        START_WORKER=True,
    )
    if config:
        app.config.update(config)
    if not 0.1 <= app.config['SAMPLE_SECONDS'] <= 10 or not 0.01 <= app.config['MODEL_CONFIDENCE'] <= 0.99:
        raise ValueError('SAMPLE_SECONDS: 0.1–10; MODEL_CONFIDENCE: 0.01–0.99 болуы тиіс.')
    if not 1 <= app.config['TEMPORAL_MIN_HITS'] <= 10 or not 0 < app.config['TEMPORAL_IOU'] <= 1:
        raise ValueError('TEMPORAL_MIN_HITS: 1–10; TEMPORAL_IOU: (0,1] болуы тиіс.')
    root = Path(app.config['STORAGE'])
    app.config['STORAGE'] = root
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ('raw', 'evidence', 'reports', 'reviewed', 'images'):
        (root / name).mkdir(exist_ok=True, mode=0o700)
    if not app.config['SECRET_KEY']:
        secret_file = root / 'session.key'
        if not secret_file.exists():
            secret_file.write_text(secrets.token_hex(32))
            secret_file.chmod(0o600)
        app.config['SECRET_KEY'] = secret_file.read_text().strip()
    password = app.config.get('ADMIN_PASSWORD') or os.getenv('ADMIN_PASSWORD')
    if not password:
        password_file = root / 'admin-password.txt'
        if not password_file.exists():
            password_file.write_text(secrets.token_urlsafe(18))
            password_file.chmod(0o600)
        password = password_file.read_text().strip()
    app.config['ADMIN_PASSWORD_HASH'] = generate_password_hash(password)
    db.init_app(app)
    app.extensions['road_detector'] = RoadDetector(app.config)
    with app.app_context():
        db.create_all()
        migrate_early_schema()
        db.session.execute(db.text('PRAGMA journal_mode=WAL'))
        if app.config['START_WORKER']:
            # Бір процесс: қайта іске қосылғанда аяқталмаған жұмыс анық көрсетіледі.
            ProcessingJob.query.filter(ProcessingJob.status.in_(['queued', 'processing'])).update(
                {'status': 'failed', 'error': 'Сервер қайта іске қосылды. Бейнені қайта жүктеңіз.'})
        db.session.commit()
    from .jobs import JobRunner
    app.extensions['job_runner'] = JobRunner(app)
    from .routes import bp
    app.register_blueprint(bp)
    from .mapping import bp as mapping_bp
    app.register_blueprint(mapping_bp)
    app.config['mapping'] = True
    from .monitoring import bp as monitoring_bp
    app.register_blueprint(monitoring_bp)
    app.config['transparency'] = True
    from .vibration import bp as vibration_bp
    app.register_blueprint(vibration_bp)
    from .reviews import bp as reviews_bp, REASONS
    app.register_blueprint(reviews_bp)
    from .images import bp as images_bp
    app.register_blueprint(images_bp)
    from .dataset import bp as dataset_bp
    app.register_blueprint(dataset_bp)

    @app.context_processor
    def context():
        from .routes import csrf_token
        from .presets import PRESETS
        return dict(labels=LABELS, csrf_token=csrf_token, review_reasons=REASONS, presets=PRESETS,
                    statuses={'queued': 'Кезекте', 'processing': 'Талдануда', 'completed': 'Аяқталды',
                              'failed': 'Қате', 'predicted': 'Модель болжамы', 'confirmed': 'Расталған',
                              'rejected': 'Қабылданбаған'})

    @app.after_request
    def security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https://*.tile.openstreetmap.org; font-src 'self'; "
            "media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
        if not response.headers.get('Cache-Control') or request_is_private():
            response.headers['Cache-Control'] = 'private, no-store'
        return response

    @app.errorhandler(413)
    def too_large(_):
        return render_template('error.html', message='Файл рұқсат етілген көлемнен үлкен.'), 413

    @app.errorhandler(404)
    def not_found(_):
        return render_template('error.html', message='Жазба табылмады.'), 404

    @app.errorhandler(400)
    def invalid(_):
        return render_template('error.html', message='Сұрау жарамсыз. Бетті жаңартып, қайта көріңіз.'), 400

    @app.errorhandler(500)
    def server_error(_):
        db.session.rollback()
        return render_template('error.html', message='Ішкі қате. Сервер журналын тексеріңіз.'), 500

    @app.get('/health')
    def health():
        return jsonify(status='ok', model_loaded=app.extensions['road_detector'].model is not None)

    return app


def request_is_private():
    from flask import request
    return not request.path.startswith('/static/')
