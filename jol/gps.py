import csv
import io
import math
from bisect import bisect_left
from datetime import datetime, timezone
from xml.etree import ElementTree


def coordinates(latitude, longitude):
    lat, lon = float(latitude), float(longitude)
    if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError('Координаттар диапазоны жарамсыз.')
    return lat, lon


def track_time(value, observed_at):
    try:
        result = float(value)
    except ValueError:
        date = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if date.tzinfo is None:
            raise ValueError('GPS күн-уақытында UTC ығысуы қажет: +05:00 немесе Z.')
        result = (date.astimezone(timezone.utc).replace(tzinfo=None) - observed_at).total_seconds()
    if not math.isfinite(result):
        raise ValueError('GPS уақыты жарамсыз.')
    return result


def parse_track(contents, extension, observed_at):
    if len(contents) > 2 * 1024 * 1024:
        raise ValueError('GPS файлы 2 МБ-тан аспауы тиіс.')
    text = contents.decode('utf-8-sig')
    if extension == '.csv':
        reader = csv.DictReader(io.StringIO(text))
        if not {'timestamp', 'latitude', 'longitude'} <= set(reader.fieldnames or []):
            raise ValueError('CSV бағандары: timestamp,latitude,longitude.')
        rows = list(reader)
    elif extension == '.gpx':
        if '<!DOCTYPE' in text.upper() or '<!ENTITY' in text.upper():
            raise ValueError('GPX ішіндегі DTD/ENTITY рұқсат етілмейді.')
        root = ElementTree.fromstring(text)
        rows = [{'timestamp': node.findtext('{*}time', ''),
                 'latitude': node.get('lat'), 'longitude': node.get('lon')}
                for node in root.findall('.//{*}trkpt')]
    else:
        raise ValueError('CSV немесе GPX файлын таңдаңыз.')
    if not rows or len(rows) > 20000:
        raise ValueError('GPS файлы бос немесе 20 000 нүктеден артық.')
    points = []
    for row in rows:
        lat, lon = coordinates(row['latitude'], row['longitude'])
        points.append((track_time(row['timestamp'], observed_at), lat, lon))
    points.sort()
    if len({point[0] for point in points}) != len(points):
        raise ValueError('GPS уақыт белгілері қайталанбауы тиіс.')
    return points


def synchronize(points, video_seconds, offset=0, max_gap=5):
    """track_seconds = video_seconds + offset; аралық >5с болса интерполяция жоқ."""
    if not points:
        return None
    target = video_seconds + offset
    times = [p[0] for p in points]
    index = bisect_left(times, target)
    if index < len(points) and abs(points[index][0] - target) < 1e-6:
        return points[index][1:]
    if index == 0 or index == len(points):
        return None
    before, after = points[index - 1], points[index]
    if after[0] - before[0] > max_gap:
        return None
    ratio = (target - before[0]) / (after[0] - before[0])
    return (before[1] + ratio * (after[1] - before[1]),
            before[2] + ratio * (after[2] - before[2]))
