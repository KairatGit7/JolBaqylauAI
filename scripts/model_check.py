"""Іске қосу: .venv/bin/python -m scripts.model_check [--image path.jpg]"""
import argparse
import json
import cv2
import numpy as np
import torch
from jol import create_app
from jol.vision import annotate

parser = argparse.ArgumentParser(description='Нақты RDD2022 моделін тексеру')
parser.add_argument('--image')
parser.add_argument('--confidence', type=float, help='Әдепкісі MODEL_CONFIDENCE (0.60).')
parser.add_argument('--output', default='artifacts/model-check.jpg')
args = parser.parse_args()
app = create_app({'START_WORKER': False})
detector = app.extensions['road_detector']
model = detector.load()
frame = cv2.imread(args.image) if args.image else np.zeros((640, 640, 3), dtype=np.uint8)
if frame is None:
    raise SystemExit('Сурет ашылмады.')
results = detector.predict(frame, args.confidence)
from pathlib import Path
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(args.output, annotate(frame, results, 0))
print(json.dumps({'model': app.config['MODEL_REPO'], 'revision': app.config['MODEL_REVISION'],
                  'confidence_threshold': args.confidence if args.confidence is not None else app.config['MODEL_CONFIDENCE'],
                  'classes': model.names, 'cuda_available': torch.cuda.is_available(),
                  'device': app.config['MODEL_DEVICE'], 'image': args.image or 'synthetic_black_frame',
                  'predictions': results}, ensure_ascii=False, indent=2))
