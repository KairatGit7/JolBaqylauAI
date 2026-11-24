"""Тек жарияланған бағдарламалық ресурстар мен лицензияланған RDD2022 үлгілері."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def download(url, relative):
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        contents = response.read()
    target.write_bytes(contents)
    return {'file': relative, 'url': url, 'sha256': hashlib.sha256(contents).hexdigest()}


if __name__ == '__main__':
    records = []
    for name in ['leaflet.css', 'leaflet.js', 'images/layers.png', 'images/layers-2x.png',
                 'images/marker-icon.png', 'images/marker-icon-2x.png', 'images/marker-shadow.png']:
        records.append(download('https://unpkg.com/leaflet@1.9.4/dist/' + name, 'jol/static/vendor/leaflet/' + name))
    records.append(download('https://raw.githubusercontent.com/Leaflet/Leaflet/v1.9.4/LICENSE', 'jol/static/vendor/leaflet/LICENSE'))
    records.append(download('https://raw.githubusercontent.com/dejavu-fonts/dejavu-fonts/master/LICENSE', 'jol/static/fonts/LICENSE'))
    for name in ['China_Drone_000008', 'China_Drone_000099', 'China_Drone_000101', 'China_Drone_000116']:
        for directory, extension in [('images', 'jpg'), ('labels', 'txt')]:
            url = f'https://huggingface.co/datasets/dronefreak/RDD2022/resolve/main/data/{directory}/test/shard_000/{name}.{extension}'
            records.append(download(url, f'samples/rdd2022/{name}.{extension}'))
    (ROOT / 'samples/sources.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Расталған ресурстар сақталды: {len(records)}')
