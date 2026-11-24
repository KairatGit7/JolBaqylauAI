"""Нақты RDD2022 моделі. Жалған болжамға ауысу қарастырылмаған."""
import math
import os
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LABELS = {
    'longitudinal_crack': 'Бойлық жарықшақ',
    'transverse_crack': 'Көлденең жарықшақ',
    'alligator_crack': 'Торлы жарықшақ',
    'pothole': 'Ойық',
}
ALIASES = {'D00': 'longitudinal_crack', 'D10': 'transverse_crack',
           'D20': 'alligator_crack', 'D40': 'pothole'}
REVISION = 'acf216a43d796f7910fc854f662c1e9cfa03fc01'


def category_name(name):
    key = ALIASES.get(name, name.lower().replace(' ', '_'))
    if key not in LABELS:
        raise ValueError(f'Модельдің белгісіз санаты: {name}')
    return key


class RoadDetector:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.lock = threading.Lock()

    def load(self):
        with self.lock:
            if self.model is None:
                import torch
                from huggingface_hub import hf_hub_download
                from ultralytics import YOLO
                device = self.config['MODEL_DEVICE']
                if device != 'cpu' and not torch.cuda.is_available():
                    raise RuntimeError('CUDA қолжетімсіз. MODEL_DEVICE=cpu орнатыңыз.')
                torch.set_num_threads(min(4, os.cpu_count() or 1))
                weights = hf_hub_download(repo_id=self.config['MODEL_REPO'],
                                          filename=self.config['MODEL_FILENAME'],
                                          revision=self.config['MODEL_REVISION'])
                model = YOLO(weights, task='detect')
                classes = {category_name(n) for n in model.names.values()}
                if classes != set(LABELS):
                    raise ValueError('Бұл модель төрт RDD2022 санатына сәйкес келмейді.')
                self.model = model
        return self.model

    def predict(self, frame, confidence=None, *, imgsz=640):
        confidence = self.config['MODEL_CONFIDENCE'] if confidence is None else confidence
        if not math.isfinite(confidence) or not 0.01 <= confidence <= 0.99:
            raise ValueError('Сенімділік шегі 0.01–0.99 аралығында болуы тиіс.')
        if imgsz not in (640, 960):
            raise ValueError('Талдау өлшемі 640 немесе 960 болуы тиіс.')
        model = self.load()
        with self.lock:
            result = model.predict(frame, conf=confidence, imgsz=imgsz, classes=None,
                                   device=self.config['MODEL_DEVICE'], verbose=False)[0]
        return [{'category': category_name(model.names[int(box.cls.item())]),
                 'confidence': float(box.conf.item()),
                 'bbox': [round(float(v), 2) for v in box.xyxy[0].tolist()]}
                for box in result.boxes]


def inspect_video(path, max_seconds=180):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ok, frame = cap.read()
        if not ok or not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count <= 0:
            raise ValueError('Бейне ашылмады немесе кодек қолдау таппады.')
        duration = count / fps
        if duration > max_seconds + 0.05:
            raise ValueError(f'Бейне ұзақтығы {max_seconds} секундтан аспауы тиіс.')
        h, w = frame.shape[:2]
        if w * h > 3840 * 2160 or fps > 120:
            raise ValueError('Ең көбі 4K және 120 кадр/с қолдау табады.')
        return dict(duration=duration, fps=fps, width=w, height=h)
    finally:
        cap.release()


def sampled_frames(path, sample_seconds, max_seconds=180):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Бейнені декодтау мүмкін болмады.')
        step = max(1, round(fps * sample_seconds))
        index = 0
        while cap.grab():
            timestamp = index / fps
            if timestamp >= max_seconds:
                break
            if index % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError('Бейне кадры бүлінген.')
                yield index, timestamp, frame
            index += 1
    finally:
        cap.release()


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / (area - intersection) if area > intersection else 0.0


class EventAggregator:
    """Бір кадрда бір оқиғаға бір ғана қорап; уақыт + IoU эвристикасы."""
    def __init__(self, max_gap=2.1, min_iou=0.2, min_hits=1, duplicate_iou=None):
        self.events = []
        self.max_gap = max_gap
        self.min_iou = min_iou
        self.min_hits = min_hits
        self.duplicate_iou = duplicate_iou
        self.last_timestamp = None

    def add(self, predictions, timestamp, evidence, video_id):
        previous_timestamp = self.last_timestamp
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError('Кадр уақыты қатаң өсуі тиіс.')
        self.last_timestamp = timestamp
        if self.duplicate_iou is not None:
            predictions = suppress_duplicates(predictions, self.duplicate_iou)
        used = set()
        for detection in sorted(predictions, key=lambda d: d['confidence'], reverse=True):
            candidates = [(iou(event['last_bbox'], detection['bbox']), i)
                          for i, event in enumerate(self.events)
                           if i not in used and event['category'] == detection['category']
                           and (self.min_hits == 1 or event['end_seconds'] == previous_timestamp)
                          and 0 < timestamp - event['end_seconds'] <= self.max_gap]
            score, index = max(candidates, default=(0, -1))
            hit = dict(detection, timestamp=timestamp, video_id=video_id)
            if score >= self.min_iou:
                event = self.events[index]
                event['end_seconds'] = timestamp
                event['last_bbox'] = detection['bbox']
                event['hits'].append(hit)
                if detection['confidence'] > event['confidence']:
                    event.update(confidence=detection['confidence'], bbox=detection['bbox'], evidence_name=evidence)
            else:
                index = len(self.events)
                self.events.append(dict(detection, start_seconds=timestamp, end_seconds=timestamp,
                                        evidence_name=evidence, hits=[hit], last_bbox=detection['bbox']))
            used.add(index)

    def results(self):
        for event in self.events:
            yield {key: value for key, value in event.items() if key != 'last_bbox'} | {
                'filter_status': 'passed' if len(event['hits']) >= self.min_hits else 'suppressed'}


def suppress_duplicates(predictions, threshold=0.5):
    """Class-aware frame NMS. Different damage classes are not silently merged."""
    kept = []
    for detection in sorted(predictions, key=lambda item: item['confidence'], reverse=True):
        if not any(item['category'] == detection['category'] and iou(item['bbox'], detection['bbox']) >= threshold
                   for item in kept):
            kept.append(detection)
    return kept


def font_path():
    options = [Path(__file__).parent / 'static/fonts/DejaVuSans.ttf',
               Path('/usr/share/fonts/TTF/DejaVuSans.ttf'),
               Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    return next((str(p) for p in options if p.exists()), None)


def annotate(frame, predictions, timestamp=None):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path(), max(15, image.width // 55)) if font_path() else ImageFont.load_default()
    for item in predictions:
        box = tuple(item['bbox'])
        draw.rectangle(box, outline='#19e4cc', width=max(2, image.width // 350))
        text = f"{LABELS[item['category']]} · {item['confidence']:.0%}"
        xy = (max(0, box[0]), max(0, box[1] - font.size - 8))
        draw.rectangle(draw.textbbox(xy, text, font=font), fill='#102538')
        draw.text(xy, text, fill='white', font=font)
    caption = 'ЖолБақылау AI · Модель болжамы' if timestamp is None else f'ЖолБақылау AI · {timestamp:.1f} с · Модель болжамы'
    draw.text((12, 12), caption, fill='#19e4cc', font=font,
              stroke_width=1, stroke_fill='#102538')
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
