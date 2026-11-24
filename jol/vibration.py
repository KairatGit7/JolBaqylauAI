import csv
import io
import math
from statistics import median

from flask import Blueprint, flash, render_template, request
from .gps import coordinates
from .routes import login_required

bp = Blueprint('vibration', __name__)


def analyze_vibration(contents):
    if len(contents) > 2 * 1024 * 1024:
        raise ValueError('Файл 2 МБ-тан аспауы тиіс.')
    reader = csv.DictReader(io.StringIO(contents.decode('utf-8-sig')))
    required = {'timestamp', 'acceleration_x', 'acceleration_y', 'acceleration_z'}
    if not required <= set(reader.fieldnames or []):
        raise ValueError('timestamp, acceleration_x, acceleration_y, acceleration_z бағандары қажет.')
    rows = []
    for row in reader:
        t, ax, ay, az = [float(row[key]) for key in ['timestamp', 'acceleration_x', 'acceleration_y', 'acceleration_z']]
        if not all(math.isfinite(v) and abs(v) < 1e7 for v in (t, ax, ay, az)):
            raise ValueError('Өлшемдер ақырлы сан болуы тиіс.')
        lat, lon = (coordinates(row['latitude'], row['longitude']) if row.get('latitude') and row.get('longitude') else (None, None))
        speed = float(row['speed']) if row.get('speed') else None
        if speed is not None and (not math.isfinite(speed) or speed < 0):
            raise ValueError('Жылдамдық жарамсыз.')
        rows.append(dict(timestamp=t, magnitude=math.sqrt(ax*ax+ay*ay+az*az), latitude=lat, longitude=lon, speed=speed))
        if len(rows) > 20000:
            raise ValueError('Ең көбі 20 000 өлшем.')
    if len(rows) < 10:
        raise ValueError('Кемінде 10 өлшем қажет.')
    rows.sort(key=lambda r: r['timestamp'])
    if len({r['timestamp'] for r in rows}) != len(rows):
        raise ValueError('Уақыт белгілері қайталанбауы тиіс.')
    center = median(r['magnitude'] for r in rows)
    mad = median(abs(r['magnitude'] - center) for r in rows)
    threshold = center + 6 * max(1.4826 * mad, 0.2)
    spikes = [r for r in rows if r['magnitude'] > threshold]
    low, high = rows[0]['timestamp'], rows[-1]['timestamp']
    maximum = max(max(r['magnitude'] for r in rows), threshold) * 1.05
    # Көрсету үшін әр терезедегі ең жоғары мән сақталады: шыңдар жоғалмайды.
    step = max(1, math.ceil(len(rows) / 800))
    display = [max(rows[i:i+step], key=lambda r: r['magnitude']) for i in range(0, len(rows), step)]
    points = ' '.join(f"{25+(r['timestamp']-low)/(high-low)*850:.2f},{235-r['magnitude']/maximum*205:.2f}" for r in display)
    return dict(rows=len(rows), threshold=threshold, spikes=spikes, points=points,
                threshold_y=235-threshold/maximum*205, low=low, high=high, maximum=maximum)


@bp.route('/vibration', methods=['GET', 'POST'])
@login_required
def vibration_page():
    result = None
    status = 200
    if request.method == 'POST':
        try:
            upload = request.files.get('sensor')
            if not upload or not (upload.filename or '').lower().endswith('.csv'):
                raise ValueError('CSV файлын таңдаңыз.')
            result = analyze_vibration(upload.stream.read(2*1024*1024+1))
        except (ValueError, UnicodeError, KeyError, TypeError) as exc:
            flash(str(exc), 'error')
            status = 400
    return render_template('vibration.html', result=result), status
