"""Жариялауға арналған КӨШІРМЕДЕ қолмен белгіленген бет/нөмір аумақтарын бүркемелеу."""
import argparse
import json
from pathlib import Path
import cv2

parser = argparse.ArgumentParser(description='Қолмен берілген қораптар бойынша бүркемелеу')
parser.add_argument('image', type=Path)
parser.add_argument('rectangles', type=Path, help='JSON: [[x1,y1,x2,y2], ...]')
parser.add_argument('output', type=Path)
args = parser.parse_args()
if args.output.resolve() == args.image.resolve():
    raise SystemExit('Түпнұсқаны алмастырмай, бөлек файлды көрсетіңіз.')
frame = cv2.imread(str(args.image))
if frame is None:
    raise SystemExit('Сурет ашылмады.')
boxes = json.loads(args.rectangles.read_text())
for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
        raise SystemExit('Қорап сурет шекарасында болуы тиіс.')
    # Қара тұтас жабу: әлсіз бұлыңғырлау арқылы деректің қалпына келуінен сақтайды.
    frame[y1:y2, x1:x2] = 0
args.output.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(args.output), frame):
    raise SystemExit('Көшірмені сақтау мүмкін болмады.')
print('Бүркемеленген көшірме:', args.output)
