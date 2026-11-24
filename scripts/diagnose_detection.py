"""Offline diagnostic matrix using the existing checkpoint; never changes app defaults."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

import cv2
from PIL import Image, ImageDraw

from jol import create_app
from jol.vision import EventAggregator, annotate, sampled_frames

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [.10, .15, .25, .40, .60]
SIZES = [640, 960]


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def get_frame(path, index):
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'Frame cannot be decoded: {index}')
        return frame
    finally:
        cap.release()


def temporal(rows, confidence, size, sample_step, fps):
    tracker = EventAggregator(max_gap=sample_step / fps * 1.1, min_hits=2, min_iou=.2, duplicate_iou=.5)
    selected = [row for row in rows if row['confidence'] == confidence and row['imgsz'] == size
                and row['frame_index'] % sample_step == 0]
    for row in selected:
        tracker.add(row['predictions'], row['frame_index'] / fps, '', 'diagnostic')
    events = list(tracker.results())
    return dict(frames=len(selected), raw_boxes=sum(len(row['predictions']) for row in selected),
                events=len(events), passed_events=sum(e['filter_status'] == 'passed' for e in events),
                suppressed_events=sum(e['filter_status'] == 'suppressed' for e in events))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/private/diagnostics-2026-11-24')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    connection = sqlite3.connect(ROOT / 'instance/jol.db')
    connection.row_factory = sqlite3.Row
    videos = {name: dict(connection.execute('SELECT * FROM video WHERE name=? ORDER BY created_at DESC LIMIT 1',
                                           (name,)).fetchone()) for name in ('roadtest.mp4', 'roadtest1.mp4')}
    connection.close()
    with tempfile.TemporaryDirectory(prefix='diagnostic-', dir='/tmp/opencode') as temporary:
        app = create_app({'START_WORKER': False, 'STORAGE': Path(temporary),
                          'SQLALCHEMY_DATABASE_URI': f'sqlite:///{temporary}/test.db',
                          'ADMIN_PASSWORD': 'diagnostic-only', 'SECRET_KEY': 'diagnostic-only'})
        detector = app.extensions['road_detector']
        model = detector.load()
        weights = Path(model.ckpt_path)
        network_shapes = []
        hook = model.model.register_forward_pre_hook(lambda module, inputs: network_shapes.append(list(inputs[0].shape)))
        latest = videos['roadtest1.mp4']
        path = ROOT / 'instance/raw' / latest['stored_name']
        scan = []
        for index, timestamp, frame in sampled_frames(path, 1):
            scan.append(dict(frame_index=index, timestamp=timestamp, predictions=detector.predict(frame, .10)))
        (out / 'scan-360p-before-application-filter.json').write_text(json.dumps(scan, indent=2))
        ranked = sorted(scan, key=lambda row: max((p['confidence'] for p in row['predictions'] if p['category'] == 'pothole'), default=0), reverse=True)
        representative_indices = sorted({scan[i]['frame_index'] for i in [0, len(scan)//3, 2*len(scan)//3, len(scan)-1]}
                                        | {row['frame_index'] for row in ranked[:4]})
        sheet = Image.new('RGB', (960, 290 * ((len(representative_indices)+1)//2)), 'white')
        draw = ImageDraw.Draw(sheet)
        inputs = []
        for offset, index in enumerate(representative_indices):
            frame = get_frame(path, index)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((480, 270))
            x, y = offset % 2 * 480, offset // 2 * 290
            sheet.paste(image, (x, y+20))
            draw.text((x+5, y+3), f'roadtest1 frame {index}, {index/latest["fps"]:.3f}s', fill='black')
            inputs.append((f'roadtest1-{index}', frame))
        sheet.save(out / 'representative-original-frames.jpg')
        for image_path in sorted((ROOT / 'samples/rdd2022').glob('*.jpg')):
            inputs.append((image_path.stem, cv2.imread(str(image_path))))
        for index in (1080, 1110, 1140):
            inputs.append((f'manhole-{index}', get_frame(ROOT / 'instance/raw' / videos['roadtest.mp4']['stored_name'], index)))

        def predict(frame, confidence, size):
            predictions = detector.predict(frame, confidence, imgsz=size)
            return dict(confidence=confidence, imgsz=size, original_shape=list(frame.shape),
                        network_shape=network_shapes[-1], class_filter=model.predictor.args.classes,
                        predictions=predictions)

        matrix = []
        for name, frame in inputs:
            for size in SIZES:
                for confidence in THRESHOLDS:
                    record = dict(source=name, **predict(frame, confidence, size))
                    matrix.append(record)
                    cv2.imwrite(str(out / f'{name}-size{size}-conf{confidence:.2f}.jpg'), annotate(frame, record['predictions']))
        (out / 'comparison-raw-predictions.json').write_text(json.dumps(matrix, ensure_ascii=False, indent=2))

        # Every original frame in one two-second interval, not the one-second application sample.
        center = ranked[0]['frame_index']
        fps = latest['fps']
        start = max(0, center - round(fps))
        end = min(round(latest['duration'] * fps), start + round(2 * fps))
        segment = []
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for index in range(start, end):
            ok, frame = cap.read()
            if not ok:
                break
            for size in SIZES:
                for confidence in THRESHOLDS:
                    segment.append(dict(frame_index=index, **predict(frame, confidence, size)))
        cap.release()
        hook.remove()
        (out / 'every-frame-raw-predictions.json').write_text(json.dumps(segment, ensure_ascii=False, indent=2))
        summaries = []
        for size in SIZES:
            for confidence in THRESHOLDS:
                summaries.append(dict(imgsz=size, confidence=confidence,
                    every_frame=temporal(segment, confidence, size, 1, fps),
                    application_sampling=temporal(segment, confidence, size, round(fps), fps)))
        report = dict(model=dict(repo=app.config['MODEL_REPO'], revision=app.config['MODEL_REVISION'],
                      weights_file=weights.name, weights_bytes=weights.stat().st_size, sha256=digest(weights),
                      classes=model.names, task=model.task, device=app.config['MODEL_DEVICE']),
                      existing_settings=dict(confidence=app.config['MODEL_CONFIDENCE'], imgsz=640,
                          sample_seconds=app.config['SAMPLE_SECONDS'], min_hits=app.config['TEMPORAL_MIN_HITS'],
                          temporal_iou=app.config['TEMPORAL_IOU']),
                      input_video=dict(name=latest['name'], sha256=digest(path), fps=fps,
                          width=latest['width'], height=latest['height'], frames_scanned=len(scan)),
                      representative_frames=representative_indices, comparison_images=len(matrix),
                      interval=dict(start_frame=start, end_frame_exclusive=end, frames=end-start,
                                    start_seconds=start/fps, end_seconds=end/fps),
                      summary=summaries,
                      interpretation='Raw means YOLO confidence/NMS output before application NMS, temporal filtering or human review, not pre-NMS logits. Temporal results are separate offline replays. No application settings or database predictions changed.',
                      limitations=['Representative frames selected partly by low-threshold detections, not unbiased evaluation.',
                                   'No new video ground-truth labels; counts are not precision/recall or accuracy.',
                                   'Upscaling 360p to 960 does not restore missing optical detail.'])
        (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
