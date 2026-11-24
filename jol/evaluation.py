"""IoU бойынша бір-біріне сәйкестендірілген объект детекциясының метрикалары."""
from .vision import LABELS, iou


def match_boxes(predictions, truth, threshold=0.5):
    used = set()
    tp = fp = 0
    for prediction in sorted(predictions, key=lambda p: p['confidence'], reverse=True):
        candidates = [(iou(prediction['bbox'], target['bbox']), index)
                      for index, target in enumerate(truth)
                      if index not in used and target['category'] == prediction['category']]
        score, index = max(candidates, default=(0, -1))
        if score >= threshold and index >= 0:
            tp += 1
            used.add(index)
        else:
            fp += 1
    return tp, fp, len(truth) - len(used)


def metrics(tp, fp, fn):
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def evaluate(records, threshold=0.5):
    if not 0 < threshold <= 1:
        raise ValueError('IoU шегі (0,1] аралығында болуы тиіс.')
    totals = [0, 0, 0]
    per_class = {key: [0, 0, 0] for key in LABELS}
    for record in records:
        counts = match_boxes(record['predictions'], record['truth'], threshold)
        totals = [a + b for a, b in zip(totals, counts)]
        for key in LABELS:
            counts = match_boxes([d for d in record['predictions'] if d['category'] == key],
                                 [d for d in record['truth'] if d['category'] == key], threshold)
            per_class[key] = [a + b for a, b in zip(per_class[key], counts)]
    return {'images': len(records), 'iou_threshold': threshold, **metrics(*totals),
            'per_class': {key: metrics(*counts) for key, counts in per_class.items()}}
