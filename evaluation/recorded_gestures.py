"""Save small, reproducible real-model gesture regression traces from paired replay."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / 'report.json').read_text(encoding='utf-8'))
    episode = next(item for item in report['episodes'] if item['file'] == 'eli_pjump.avi')
    if not episode['eligible']:
        raise ValueError('Recorded reference must pass alignment before saving a regression fixture')
    annotations = Path(__file__).resolve().parent / 'annotations/weizmann-jumps.json'
    labels = json.loads(annotations.read_text(encoding='utf-8'))['clips']['eli_pjump.avi']
    if report['annotation_sha256'] != hashlib.sha256(annotations.read_bytes()).hexdigest():
        raise ValueError('Replay was produced using different annotations; regenerate it first')
    clips = []
    for key, kind in (('motion_result', 'jump'), ('reference_result', 'standing_arm_motion')):
        result = episode[key]
        trace = args.run / (Path(result['file']).stem + '.jsonl')
        rows = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
        # For the negative case retain only actual observations after successful calibration.
        if key == 'reference_result':
            rows = [row for row in rows if row['calibrated']]
        clips.append({'file': result['file'], 'source_sha256': result['sha256'],
                      'kind': kind, 'fps': result['fps'],
                      'events': labels['events'] if key == 'motion_result' else [],
                      'rows': [{'timestamp_ms': row['timestamp_ms'], 'features': row['features']}
                               for row in rows]})
    fixture = {'dataset': 'Weizmann, development subject Eli',
               'source_page': 'https://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html',
               'scope': 'Two visually reviewed jump cycles and one short standing arm-motion segment; no live game accuracy claim',
               'model_sha256': report['model_sha256'], 'source_sha256': report['source_sha256'],
               'profile': episode['reference_result']['calibration_profile'], 'clips': clips}
    args.output.write_text(json.dumps(fixture, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
