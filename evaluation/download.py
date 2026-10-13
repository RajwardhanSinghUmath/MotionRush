"""Download original videos or archive members and verify source/content hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import ssl
from urllib.request import Request, urlopen
import zipfile

import certifi


DEFAULT_MANIFEST = Path(__file__).with_name('ucf101_manifest.json')


def safe_filename(name):
    if (Path(name).name != name or name in ('.', '..')
            or any(character in name for character in ('/', '\\', ':', '\x00'))):
        raise ValueError('Video filename must not contain directories')
    return name


def verify(content, metadata):
    if len(content) != metadata['bytes'] or hashlib.sha256(content).hexdigest() != metadata['sha256']:
        raise ValueError('Source size or checksum mismatch: ' + metadata['file'])


def write_verified(content, metadata, target):
    verify(content, metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.part')
    temporary.write_bytes(content)
    temporary.replace(target)


def download_source(source, target):
    if target.exists():
        content = target.read_bytes()
    else:
        request = Request(source['source_url'], headers={'User-Agent': 'MotionRush-evaluation'})
        context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(request, timeout=45, context=context) as response:
            content = response.read(source['bytes'] + 1)
    verify(content, source)
    if not target.exists():
        write_verified(content, source, target)
    return target


def download_clip(clip, destination):
    name = safe_filename(clip['file'])
    target = Path(destination) / name
    if target.exists():
        verify(target.read_bytes(), clip)
    elif 'archive' in clip:
        source = clip['archive']
        archive_path = Path(destination) / '.archives' / safe_filename(source['file'])
        download_source(source, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            info = archive.getinfo(clip['archive_member'])
            if info.file_size != clip['bytes']:
                raise ValueError('Archive member size mismatch: ' + name)
            # Read the member as data; never extract archive paths to disk.
            content = archive.read(info)
        write_verified(content, clip, target)
    else:
        download_source(clip, target)
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
