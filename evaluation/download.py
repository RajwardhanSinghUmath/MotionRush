"""Download the manifest's original UCF101 videos and verify their hashes."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_MANIFEST = Path(__file__).with_name('ucf101_manifest.json')


def download_clip(clip, destination):
    name = clip['file']
    if Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('Video filename must not contain directories')
    target = Path(destination) / name
    if target.exists():
        content = target.read_bytes()
    else:
        request = Request(clip['source_url'], headers={'User-Agent': 'MotionRush-evaluation'})
        with urlopen(request, timeout=45) as response:
            content = response.read(clip['bytes'] + 1)
    if len(content) != clip['bytes'] or hashlib.sha256(content).hexdigest() != clip['sha256']:
        raise ValueError('Video size or checksum mismatch: ' + name)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.part')
        temporary.write_bytes(content)
        temporary.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--destination', type=Path, default=Path('evaluation/data'))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    for clip in manifest['clips']:
        print(download_clip(clip, args.destination), flush=True)


if __name__ == '__main__':
    main()
