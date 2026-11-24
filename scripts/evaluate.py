"""YOLO таңбалары: image.jpg + image.txt; класс cx cy w h, [0,1]."""
import argparse
import json
import math
from pathlib import Path
import cv2

from jol import create_app
from jol.evaluation import evaluate
from jol.vision import LABELS


def read_truth(path, width, height):
    result = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f'{path}:{number}: класс cx cy w h қажет.')
        category = int(values[0])
        cx, cy, w, h = map(float, values[1:])
        if category not in range(4) or any(not math.isfinite(x) or not 0 <= x <= 1 for x in [cx, cy, w, h]) or w <= 0 or h <= 0:
            raise ValueError(f'{path}:{number}: жарамсыз таңба.')
        box = [(cx - w/2)*width, (cy - h/2)*height, (cx + w/2)*width, (cy + h/2)*height]
        if box[0] < -1 or box[1] < -1 or box[2] > width+1 or box[3] > height+1:
            raise ValueError(f'{path}:{number}: қорап сурет шекарасынан тыс.')
        result.append(dict(category=list(LABELS)[category], bbox=box))
    return result


def main():
    parser = argparse.ArgumentParser(description='Қолмен таңбаланған суреттер бойынша бағалау')
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/evaluation.json'))
    parser.add_argument('--confidence', type=float, help='Шек; әдепкісі MODEL_CONFIDENCE (0.60).')
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--provenance', required=True, help='Дерек шығу тегі, лицензиясы, тәуелсіздігі')
    args = parser.parse_args()
    app = create_app({'START_WORKER': False})
    if args.confidence is None:
        args.confidence = app.config['MODEL_CONFIDENCE']
    if not 0 < args.confidence < 1:
        parser.error('Сенімділік (0,1) аралығында.')
    detector = app.extensions['road_detector']
    records = []
    skipped = []
    for image_path in sorted(args.directory.iterdir()):
        if image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        annotation = image_path.with_suffix('.txt')
        if not annotation.exists():
            skipped.append(image_path.name)
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f'Сурет ашылмады: {image_path}')
        truth = read_truth(annotation, image.shape[1], image.shape[0])
        records.append(dict(image=image_path.name, truth=truth, predictions=detector.predict(image, args.confidence)))
    if not records:
        raise SystemExit('Таңбаланған сурет жоқ. Өнімділік өлшенген жоқ.')
    result = evaluate(records, args.iou)
    result.update(confidence_threshold=args.confidence, model=app.config['MODEL_REPO'],
                  revision=app.config['MODEL_REVISION'], provenance=args.provenance,
                  skipped_unlabeled=skipped, records=records,
                  limitation='Шағын жиынтық Қазақстандағы сапаны растауға жеткіліксіз. Бұл mAP немесе жіктеу accuracy көрсеткіші емес.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
