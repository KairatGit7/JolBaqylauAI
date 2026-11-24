"""Paired real-model evaluation, preserving separate image/video/ROI denominators."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile

import cv2

from jol import create_app
from jol.evaluation import evaluate
from jol.vision import EventAggregator, sampled_frames
from scripts.evaluate import read_truth

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def track(records, key, *, min_hits=1, gap=2.1, nms=None):
    tracker = EventAggregator(max_gap=gap, min_iou=.2, min_hits=min_hits, duplicate_iou=nms)
    for record in records:
        tracker.add(record[key], record['timestamp'], str(record['frame_index']), 'evaluation-only')
    events = list(tracker.results())
    kept = defaultdict(list)
    for event in events:
        if event['filter_status'] == 'passed':
            for hit in event['hits']:
                kept[hit['timestamp']].append(hit)
    return events, kept


def paired_frames(detector, path):
    result = []
    for index, timestamp, frame in sampled_frames(path, 1.0):
        result.append(dict(frame_index=index, timestamp=timestamp,
                           before=detector.predict(frame, .3), threshold_only=detector.predict(frame, .6)))
    return result


def region_records(frames, annotations, stage, kept=None):
    by_index = {frame['frame_index']: frame for frame in frames}
    records = []
    for annotation in annotations:
        frame = by_index[annotation['frame_index']]
        predictions = kept[frame['timestamp']] if kept is not None else frame[stage]
        x1, y1, x2, y2 = annotation['bbox']
        selected = [item for item in predictions if item['category'] == annotation['category']
                    and x1 <= (item['bbox'][0] + item['bbox'][2]) / 2 <= x2
                    and y1 <= (item['bbox'][1] + item['bbox'][3]) / 2 <= y2]
        records.append(dict(frame_index=frame['frame_index'], kind=annotation['kind'],
                            predictions=selected, truth=annotation['truth']))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--road-video', type=Path, required=True)
    parser.add_argument('--regions', type=Path, default=ROOT / 'samples/roadtest-review-regions.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/detection-quality.json')
    args = parser.parse_args()
    regions = json.loads(args.regions.read_text())
    if sha256(args.road_video) != regions['video_sha256']:
        parser.error('Video does not match the reviewed-region SHA-256.')
    demo = ROOT / 'samples/DEMO_RDD2022_slideshow.avi'
    if sha256(demo) != '10a689ce9352eedc1dca72880a54a6e67430e84388b46b8f1ae2ec89e7b15ce7':
        parser.error('Slideshow changed; label transforms must be checked again.')
    # Source image/annotation hashes protect the known slideshow label transforms.
    for source in json.loads((ROOT / 'samples/sources.json').read_text()):
        if source['file'].startswith('samples/rdd2022/'):
            if sha256(ROOT / source['file']) != source['sha256']:
                parser.error('RDD2022 source or labels changed: ' + source['file'])
    temporary_root = Path('/tmp/opencode')
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='road-evaluation-', dir=temporary_root) as temporary:
        app = create_app({'START_WORKER': False, 'STORAGE': Path(temporary),
                          'SQLALCHEMY_DATABASE_URI': 'sqlite:///' + temporary + '/evaluation.db',
                          'SECRET_KEY': 'evaluation-only', 'ADMIN_PASSWORD': 'evaluation-only'})
        detector = app.extensions['road_detector']
        images, truths = [], []
        for path in sorted((ROOT / 'samples/rdd2022').glob('*.jpg')):
            frame = cv2.imread(str(path))
            height, width = frame.shape[:2]
            truth = read_truth(path.with_suffix('.txt'), width, height)
            images.append(dict(image=path.name, truth=truth, before=detector.predict(frame, .3),
                               threshold_only=detector.predict(frame, .6)))
            truths.append([dict(category=item['category'], bbox=[item['bbox'][0] * 640 / width,
                item['bbox'][1] * 640 / height + 40, item['bbox'][2] * 640 / width,
                item['bbox'][3] * 640 / height + 40]) for item in truth])
        image_metrics = {stage: evaluate([dict(predictions=row[stage], truth=row['truth']) for row in images])
                         for stage in ('before', 'threshold_only')}
        slideshow = paired_frames(detector, demo)
        for frame in slideshow:
            frame['truth'] = truths[frame['frame_index'] // 30]
        demo_events, demo_kept = track(slideshow, 'threshold_only', min_hits=2, gap=1.1, nms=.5)
        demo_metrics = {stage: evaluate([dict(predictions=row[stage], truth=row['truth']) for row in slideshow])
                        for stage in ('before', 'threshold_only')}
        demo_metrics['threshold_and_temporal'] = evaluate([
            dict(predictions=demo_kept[row['timestamp']], truth=row['truth']) for row in slideshow])

        road = paired_frames(detector, args.road_video)
        cap = cv2.VideoCapture(str(args.road_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        road_events, road_kept = track(road, 'threshold_only', min_hits=2,
                                       gap=max(1, round(fps)) / fps * 1.1, nms=.5)
        road_before, _ = track(road, 'before')
        road_threshold, _ = track(road, 'threshold_only')
        challenge = {}
        for kind in sorted({row['kind'] for row in regions['regions']}):
            annotations = [row for row in regions['regions'] if row['kind'] == kind]
            challenge[kind] = {stage: evaluate(region_records(road, annotations, stage, kept))
                              for stage, kept in [('before', None), ('threshold_only', None),
                                                  ('threshold_and_temporal', road_kept)]}
        result = dict(model=app.config['MODEL_REPO'], revision=app.config['MODEL_REVISION'],
                      before_confidence=.3, after_confidence=.6, temporal_min_hits=2, temporal_iou=.2,
                      frame_duplicate_iou=.5, match_iou=.5,
                      labeled_images=dict(metrics=image_metrics, records=images),
                      slideshow=dict(metrics=demo_metrics, frames=slideshow, events=demo_events),
                      road_video=dict(sha256=regions['video_sha256'], sampled_frames=len(road),
                                      annotations=regions, metrics_by_region_kind=challenge,
                                      before_events=road_before, threshold_events=road_threshold,
                                      filtered_events=road_events, frames=road),
                      limitations=[
                          '0.60 is a requested starting threshold, not a validated optimum.',
                          'Four published RDD2022 test images, six boxes; not representative local validation.',
                          'Slideshow repeats those four still images: repeated frame counts are NOT independent samples or driving-tracking validation.',
                          'Road regions are provisional assistant visual labels, not human-adjudicated; only potholes with box centers inside annotated ROIs are scored.',
                          'Unannotated areas and ambiguous asphalt patches are excluded. Recall on negative-only ROIs is undefined.',
                          'Video selected after observing the error. This is a regression diagnostic, not a held-out benchmark.',
                          'Temporal stability can retain a persistent false positive and suppress a fast-moving real defect.'
                      ])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(dict(images=image_metrics, slideshow=demo_metrics, road_regions=challenge,
                              road_events_before=len(road_before), road_events_threshold=len(road_threshold),
                              road_events_passed=sum(e['filter_status'] == 'passed' for e in road_events),
                              output=str(args.output)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
