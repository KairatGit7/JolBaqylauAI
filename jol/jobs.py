import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
from .models import db, Video, DetectionEvent, ProcessingJob, now
from .vision import EventAggregator, annotate, sampled_frames


class JobRunner:
    def __init__(self, app):
        self.app = app
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='road-vision')

    def submit(self, video_id):
        return self.pool.submit(self.process, video_id)

    def process(self, video_id):
        with self.app.app_context():
            video = db.session.get(Video, video_id)
            job = video.job
            root = self.app.config['STORAGE']
            folder = root / 'evidence' / video.id
            folder.mkdir(parents=True, exist_ok=True, mode=0o700)
            writer = None
            try:
                job.status = 'processing'
                db.session.commit()
                detector = self.app.extensions['road_detector']
                detector.load()
                aggregator = EventAggregator(max_gap=video.temporal_max_gap,
                                             min_iou=video.temporal_iou, min_hits=video.temporal_min_hits,
                                             duplicate_iou=0.5 if video.pipeline_version != 'legacy' else None)
                temporary = folder / 'sampled.mp4'
                # Браузерге арналған қысқартылған кадр жиілігі, уақыт ұзақтығы шамалас.
                scale = min(1, 1280 / video.width)
                size = (int(video.width * scale) // 2 * 2, int(video.height * scale) // 2 * 2)
                writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'),
                                         1 / video.sample_seconds, size)
                for index, seconds, frame in sampled_frames(root / 'raw' / video.stored_name,
                                                            video.sample_seconds,
                                                            self.app.config['MAX_VIDEO_SECONDS']):
                    kwargs = {'imgsz': video.inference_size} if video.inference_size != 640 else {}
                    predictions = detector.predict(frame, video.confidence, **kwargs)
                    evidence = f'{video.id}/{index:07d}.jpg'
                    rendered = annotate(frame, predictions, seconds)
                    if predictions or job.frames_analyzed == 0:
                        if not cv2.imwrite(str(root / 'evidence' / evidence), rendered):
                            raise OSError('Дәлел кадрын сақтау мүмкін болмады.')
                    if writer.isOpened():
                        writer.write(cv2.resize(rendered, size))
                    aggregator.add(predictions, seconds, evidence, video.id)
                    job.frames_analyzed += 1
                    job.progress = min(95, int(seconds / video.duration * 95))
                    db.session.commit()
                writer.release()
                writer = None
                if not job.frames_analyzed:
                    raise ValueError('Талдауға жарамды кадр жоқ.')
                if shutil.which('ffmpeg') and temporary.exists():
                    target = folder / 'annotated.mp4'
                    result = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(temporary),
                                             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                                             '-an', str(target)], capture_output=True, timeout=120)
                    if result.returncode == 0:
                        video.annotated_name = f'{video.id}/annotated.mp4'
                    else:
                        logging.warning('ffmpeg: %s', result.stderr.decode(errors='replace'))
                temporary.unlink(missing_ok=True)
                for data in aggregator.results():
                    db.session.add(DetectionEvent(video_id=video.id, **data))
                if video.sha256:
                    Video.query.filter(Video.sha256 == video.sha256, Video.owner_key == video.owner_key,
                                       Video.id != video.id).update({'is_current': False})
                video.is_current = True
                job.status = 'completed'
                job.progress = 100
                job.completed_at = now()
                db.session.commit()
            except Exception as exc:
                db.session.rollback()
                logging.exception('Бейне талдауы сәтсіз: %s', video_id)
                job = ProcessingJob.query.filter_by(video_id=video_id).one()
                job.status = 'failed'
                job.error = f'Талдау орындалмады: {type(exc).__name__}: {str(exc)[:500]}'
                job.completed_at = now()
                db.session.commit()
            finally:
                if writer is not None:
                    writer.release()
