"""Нақты RDD2022 суреттерінен айқын белгіленген слайд-бейне; бұл жүргізу бейнесі емес."""
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from jol.vision import font_path

root = Path(__file__).resolve().parents[1]
output = root / 'samples/DEMO_RDD2022_slideshow.avi'
writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'MJPG'), 10, (640, 680))
if not writer.isOpened():
    raise SystemExit('MJPG кодегі ашылмады.')
font = ImageFont.truetype(font_path(), 13)
for path in sorted((root / 'samples/rdd2022').glob('*.jpg')):
    image = Image.new('RGB', (640, 680), '#14283c')
    image.paste(Image.open(path).convert('RGB').resize((640, 640)), (0, 40))
    draw = ImageDraw.Draw(image)
    draw.text((10, 4), 'ДЕМО · RDD2022 (Қытай) · CC BY-SA 4.0 · слайд-бейне', font=font, fill='white')
    draw.text((10, 22), path.name + ' · нақты сапар / жергілікті түсірілім емес', font=font, fill='#7ce4dc')
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    for _ in range(30):
        writer.write(frame)
writer.release()
print(output)
