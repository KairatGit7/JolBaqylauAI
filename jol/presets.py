"""Explicit per-analysis presets; never mutate production configuration."""
from flask import current_app, request

PRESETS = {
    'default': dict(label='Ағымдағы баптаулар'),
    'high_recall': dict(label='Толық анықтау', confidence=.15, inference_size=640, temporal_min_hits=1),
    'balanced': dict(label='Теңгерімді', confidence=.40, inference_size=960, temporal_min_hits=2),
}


def inference_settings(fps=None):
    key = request.form.get('preset', 'default')
    if key not in PRESETS:
        raise ValueError('Тізімдегі талдау режимін таңдаңыз.')
    preset = PRESETS[key]
    confidence = preset.get('confidence', current_app.config['MODEL_CONFIDENCE'])
    if key == 'default' and fps is not None:
        confidence = float(request.form.get('confidence', confidence))
    if not .01 <= confidence <= .99:
        raise ValueError('Сенімділік шегі 0.01–0.99 аралығында болуы тиіс.')
    settings = dict(preset=key, confidence=confidence, inference_size=preset.get('inference_size', 640))
    if fps is not None:
        sample = current_app.config['SAMPLE_SECONDS'] if key == 'default' else 1 / fps
        settings.update(sample_seconds=sample,
                        temporal_min_hits=preset.get('temporal_min_hits', current_app.config['TEMPORAL_MIN_HITS']),
                        temporal_iou=current_app.config['TEMPORAL_IOU'],
                        temporal_max_gap=max(1, round(fps * sample)) / fps * 1.1)
    return settings
