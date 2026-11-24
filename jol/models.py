from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Video(db.Model):
    id = db.Column(db.String(32), primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    stored_name = db.Column(db.String(80), nullable=False)
    created_at = db.Column(db.DateTime, default=now, nullable=False)
    observed_at = db.Column(db.DateTime, nullable=False)
    duration = db.Column(db.Float, nullable=False)
    fps = db.Column(db.Float, nullable=False)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    sample_seconds = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    model_revision = db.Column(db.String(100), nullable=False)
    annotated_name = db.Column(db.String(100))
    gps_offset = db.Column(db.Float)
    gps_source = db.Column(db.String(300))
    is_demo = db.Column(db.Boolean, nullable=False, default=False, server_default='0')
    sha256 = db.Column(db.String(64), index=True)
    pipeline_version = db.Column(db.String(40), nullable=False, default='temporal-v1', server_default='legacy')
    temporal_min_hits = db.Column(db.Integer, nullable=False, default=2, server_default='1')
    temporal_iou = db.Column(db.Float, nullable=False, default=0.2, server_default='0.2')
    temporal_max_gap = db.Column(db.Float, nullable=False, default=1.1, server_default='2.1')
    is_current = db.Column(db.Boolean, nullable=False, default=True, server_default='1')
    owner_key = db.Column(db.String(64), index=True)
    preset = db.Column(db.String(30), nullable=False, default='default', server_default='default')
    inference_size = db.Column(db.Integer, nullable=False, default=640, server_default='640')
    job = db.relationship('ProcessingJob', backref='video', uselist=False)
    events = db.relationship('DetectionEvent', backref='video', lazy=True)


class ProcessingJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(32), db.ForeignKey('video.id'), unique=True, nullable=False)
    status = db.Column(db.String(20), default='queued', nullable=False)
    progress = db.Column(db.Integer, default=0, nullable=False)
    error = db.Column(db.Text)
    frames_analyzed = db.Column(db.Integer, default=0, nullable=False)
    created_at = db.Column(db.DateTime, default=now, nullable=False)
    completed_at = db.Column(db.DateTime)


class RoadSegment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False, unique=True)
    description = db.Column(db.Text, default='')
    created_at = db.Column(db.DateTime, default=now, nullable=False)


class GPSObservation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(32), db.ForeignKey('video.id'), nullable=False, index=True)
    track_seconds = db.Column(db.Float, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    video = db.relationship('Video', backref='gps_points')


def migrate_early_schema():
    """Бос MVP дерекқорынан келесі кезеңге қауіпсіз, қосымша өрістер көшіруі."""
    from sqlalchemy import inspect
    changes = {
        'video': {'gps_offset': 'FLOAT', 'gps_source': 'VARCHAR(300)',
                  'is_demo': 'BOOLEAN NOT NULL DEFAULT 0', 'sha256': 'VARCHAR(64)',
                  'pipeline_version': "VARCHAR(40) NOT NULL DEFAULT 'legacy'",
                  'temporal_min_hits': 'INTEGER NOT NULL DEFAULT 1',
                  'temporal_iou': 'FLOAT NOT NULL DEFAULT 0.2',
                  'temporal_max_gap': 'FLOAT NOT NULL DEFAULT 2.1',
                  'is_current': 'BOOLEAN NOT NULL DEFAULT 1', 'owner_key': 'VARCHAR(64)',
                  'preset': "VARCHAR(30) NOT NULL DEFAULT 'default'", 'inference_size': 'INTEGER NOT NULL DEFAULT 640'},
        'detection_event': {'segment_id': 'INTEGER REFERENCES road_segment(id)',
                             'latitude': 'FLOAT', 'longitude': 'FLOAT', 'location_source': 'VARCHAR(300)',
                             'filter_status': "VARCHAR(20) NOT NULL DEFAULT 'legacy'"},
        'image_analysis': {'preset': "VARCHAR(30) NOT NULL DEFAULT 'default'",
                           'inference_size': 'INTEGER NOT NULL DEFAULT 640'},
    }
    inspector = inspect(db.engine)
    for table, fields in changes.items():
        existing = {field['name'] for field in inspector.get_columns(table)}
        for name, definition in fields.items():
            if name not in existing:
                db.session.execute(db.text(f'ALTER TABLE {table} ADD COLUMN {name} {definition}'))
                if table == 'video' and name == 'temporal_max_gap':
                    db.session.execute(db.text('UPDATE video SET temporal_max_gap = sample_seconds * 2.1'))
    db.session.commit()


class DetectionEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(32), db.ForeignKey('video.id'), nullable=False, index=True)
    category = db.Column(db.String(40), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    start_seconds = db.Column(db.Float, nullable=False)
    end_seconds = db.Column(db.Float, nullable=False)
    bbox = db.Column(db.JSON, nullable=False)
    hits = db.Column(db.JSON, nullable=False)
    evidence_name = db.Column(db.String(100), nullable=False)
    review_status = db.Column(db.String(20), default='predicted', nullable=False)
    filter_status = db.Column(db.String(20), default='passed', server_default='legacy', nullable=False)
    review_note = db.Column(db.Text, default='')
    reviewed_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=now, nullable=False)
    segment_id = db.Column(db.Integer, db.ForeignKey('road_segment.id'), index=True)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    location_source = db.Column(db.String(300))
    segment = db.relationship('RoadSegment', backref='events')


class ImageAnalysis(db.Model):
    id = db.Column(db.String(32), primary_key=True)
    owner_key = db.Column(db.String(64), nullable=False, index=True)
    name = db.Column(db.String(200), nullable=False)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    model_revision = db.Column(db.String(100), nullable=False)
    predictions = db.Column(db.JSON, nullable=False)
    preset = db.Column(db.String(30), nullable=False, default='default', server_default='default')
    inference_size = db.Column(db.Integer, nullable=False, default=640, server_default='640')
    created_at = db.Column(db.DateTime, default=now, nullable=False)


class ReviewExample(db.Model):
    """Append-only decisions; labels concern the prediction box, not the whole image."""
    id = db.Column(db.String(32), primary_key=True)
    event_id = db.Column(db.Integer, db.ForeignKey('detection_event.id'), nullable=False, index=True)
    verdict = db.Column(db.String(20), nullable=False)
    reason = db.Column(db.String(40), nullable=False)
    note = db.Column(db.Text, nullable=False)
    reviewer = db.Column(db.String(80), nullable=False)
    image_name = db.Column(db.String(100), nullable=False)
    snapshot = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=now, nullable=False)
    event = db.relationship('DetectionEvent', backref='reviews')


class ImageReviewExample(db.Model):
    """Append-only decisions per image prediction, separate from immutable YOLO output."""
    id = db.Column(db.String(32), primary_key=True)
    image_id = db.Column(db.String(32), db.ForeignKey('image_analysis.id'), nullable=False, index=True)
    prediction_index = db.Column(db.Integer, nullable=False)
    verdict = db.Column(db.String(20), nullable=False)
    reason = db.Column(db.String(40), nullable=False)
    note = db.Column(db.Text, nullable=False)
    reviewer = db.Column(db.String(80), nullable=False)
    image_name = db.Column(db.String(100), nullable=False)
    snapshot = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=now, nullable=False)


class GroundTruthSample(db.Model):
    """Human-completed image labels, independent of predictions and box reviews."""
    id = db.Column(db.String(32), primary_key=True)
    owner_key = db.Column(db.String(64), index=True)
    source_key = db.Column(db.String(180), nullable=False)
    source_group = db.Column(db.String(100), nullable=False)
    image_id = db.Column(db.String(32), db.ForeignKey('image_analysis.id'))
    video_id = db.Column(db.String(32), db.ForeignKey('video.id'))
    frame_index = db.Column(db.Integer)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    image_hash = db.Column(db.String(64), nullable=False)
    road_group = db.Column(db.String(120), nullable=False, default='')
    split = db.Column(db.String(10), nullable=False, default='train')
    boxes = db.Column(db.JSON)  # NULL = draft, [] is valid only for an explicit negative.
    negative = db.Column(db.Boolean, nullable=False, default=False)
    revision = db.Column(db.Integer, nullable=False, default=0)
    reviewer = db.Column(db.String(80))
    updated_at = db.Column(db.DateTime, nullable=False, default=now)
    __table_args__ = (db.UniqueConstraint('owner_key', 'source_key'),)


class DetectionEvaluation(db.Model):
    id = db.Column(db.String(32), primary_key=True)
    owner_key = db.Column(db.String(64), index=True)
    snapshot = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=now)


def active_events():
    """Machine suppression is not human rejection; a reviewer can rescue a candidate."""
    return db.and_(DetectionEvent.video.has(Video.is_current.is_(True)),
                   db.or_(DetectionEvent.filter_status != 'suppressed', DetectionEvent.review_status == 'confirmed'))


class RepairContract(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    identifier = db.Column(db.String(120), unique=True, nullable=False)
    title = db.Column(db.String(300), nullable=False)
    value = db.Column(db.Numeric(18, 2), nullable=False)
    customer = db.Column(db.String(300), nullable=False)
    contractor = db.Column(db.String(300), nullable=False)
    segment_id = db.Column(db.Integer, db.ForeignKey('road_segment.id'), nullable=False, index=True)
    start_date = db.Column(db.Date)
    completion_date = db.Column(db.Date)
    warranty_end = db.Column(db.Date)
    warranty_note = db.Column(db.Text, default='')
    source_url = db.Column(db.String(1000), nullable=False)
    verification_status = db.Column(db.String(20), default='unverified', nullable=False)
    verification_note = db.Column(db.Text, default='')
    verified_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=now, nullable=False)
    segment = db.relationship('RoadSegment', backref='contracts')


class MonitoringReport(db.Model):
    id = db.Column(db.String(32), primary_key=True)
    segment_id = db.Column(db.Integer, db.ForeignKey('road_segment.id'))
    title = db.Column(db.String(300), nullable=False)
    snapshot = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=now, nullable=False)
    segment = db.relationship('RoadSegment')
