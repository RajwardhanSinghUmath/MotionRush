"""Score inspected motion clips after calibration on a separate real reference."""

import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path
from statistics import median

from control.calibration import CalibrationProfile, StandingCalibration
from config.settings import CALIBRATION_POSITION_RANGE, CALIBRATION_SCALE_RANGE
from evaluation.download import download_clip
from evaluation.replay import replay_clip, score_motion, source_provenance


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / 'evaluation/manifests/weizmann.json'
DEFAULT_ANNOTATIONS = ROOT / 'evaluation/annotations/weizmann-jumps.json'


def check_reference_alignment(profile, features):
    valid = [feature for feature in features if StandingCalibration._valid_features(feature)]
    if len(valid) < 2:
        return {'matched': False, 'reason': 'No reliable inspected initial neutral anchor'}
    x = median(feature['hip_center'][0] for feature in valid)
    y = median(feature['hip_center'][1] for feature in valid)
    torso = median(feature['torso_height'] for feature in valid)
    offsets = {'x_torso_offset': abs(x - profile.x_center) / profile.torso_height,
               'y_torso_offset': abs(y - profile.y_center) / profile.torso_height,
               'relative_scale_change': abs(torso / profile.torso_height - 1)}
    matched = (offsets['x_torso_offset'] <= CALIBRATION_POSITION_RANGE
               and offsets['y_torso_offset'] <= CALIBRATION_POSITION_RANGE
               and offsets['relative_scale_change'] <= CALIBRATION_SCALE_RANGE)
    return dict(offsets, matched=matched,
                reason='Aligned inspected anchor' if matched else 'Reference and motion clip neutral anchors differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--annotations', type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument('--data', type=Path, default=Path('evaluation/data'))
    parser.add_argument('--output', type=Path, default=Path('evaluation/runs/paired'))
    parser.add_argument('--split', choices=('development', 'validation', 'held_out'), default='development')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    labels = json.loads(args.annotations.read_text(encoding='utf-8'))
    allowed_subjects = manifest['split_policy'][args.split]
    clips = {clip['file']: clip for clip in manifest['clips']}
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        **source_provenance(),
        'model_sha256': hashlib.sha256((ROOT / 'pose_landmarker_lite.task').read_bytes()).hexdigest(),
        'annotation_sha256': hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        'split': args.split, 'subjects': allowed_subjects,
        'conditions': 'Separate real reference/motion clips; fresh center-lane episode; no keyboard injection',
        'annotation_scope': labels['description'],
        'episodes': [], 'subjects_without_annotations': [],
    }
    references = {}
    annotated_subjects = set()
    for name, label in labels['clips'].items():
        clip = clips[name]
        subject = clip['subject']
        if subject not in allowed_subjects:
            continue
        annotated_subjects.add(subject)
        reference_name = label['reference_clip']
        reference_clip = clips[reference_name]
        if reference_clip['subject'] != subject:
            raise ValueError('Reference and movement subjects must match')
        if reference_name not in references:
            path = download_clip(reference_clip, args.data)
            references[reference_name] = replay_clip(reference_clip, path, args.output)
        reference = references[reference_name]
        episode = {'file': name, 'subject': subject, 'reference_clip': reference_name,
                   'reference_result': reference}
        if reference.get('calibration_profile') is None:
            episode.update(eligible=False, reason='Production reference calibration did not complete')
        else:
            profile = CalibrationProfile(**reference['calibration_profile'])
            path = download_clip(clip, args.data)
            result = replay_clip(clip, path, args.output, reference_profile=profile)
            # Labels are inspected windows on the actual AVI playback timeline.
            events = [{'action': event['action'],
                       'start_ms': round(event['start_frame'] * 1000 / result['fps']),
                       'end_ms': round(event['end_frame'] * 1000 / result['fps'])}
                      for event in (label['events'] or [])]
            episode['motion_result'] = result
            trace_path = args.output / (Path(name).stem + '.jsonl')
            with trace_path.open(encoding='utf-8') as trace:
                prefix = [json.loads(line) for line in islice(trace, max(label['initial_neutral_frames'], default=-1) + 1)]
            initial_features = [prefix[index]['features'] for index in label['initial_neutral_frames']]
            alignment = check_reference_alignment(profile, initial_features)
            episode['reference_alignment'] = alignment
            if not alignment['matched']:
                episode.update(eligible=False, reason=alignment['reason'])
            elif label['events'] is None:
                episode.update(eligible=False, reason='No reviewed event annotations')
            else:
                episode.update(score_motion(result, events))
        report['episodes'].append(episode)
        print(name, 'eligible', episode['eligible'], episode.get('metrics', episode.get('reason')), flush=True)
    report['subjects_without_annotations'] = sorted(set(allowed_subjects) - annotated_subjects)
    report['eligible_episodes'] = sum(episode['eligible'] for episode in report['episodes'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
