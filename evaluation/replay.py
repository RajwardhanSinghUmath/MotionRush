"""Extract real poses and replay controller decisions on recorded videos."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import cv2

from control.gesture_recognizer import GestureRecognizer
from pose.estimator import PoseEstimator
from evaluation.download import DEFAULT_MANIFEST, download_clip


def source_provenance():
    root = Path(__file__).resolve().parents[1]
    modules = ('evaluation/replay.py', 'evaluation/paired_replay.py',
               'control/gesture_recognizer.py', 'control/calibration.py',
               'pose/estimator.py', 'config/settings.py')
    return {
        'revision': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in modules},
    }


def summarize_events(events, annotations):
    """Match one expected event per inclusive window; extras are false positives."""
    annotations = sorted(annotations, key=lambda item: item['start_ms'])
    last_end = {}
    for expected in annotations:
        if expected['start_ms'] > expected['end_ms']:
            raise ValueError('Annotation window ends before it starts')
        if expected['start_ms'] <= last_end.get(expected['action'], -1):
            raise ValueError('Same-action annotation windows must not overlap')
        last_end[expected['action']] = expected['end_ms']
    remaining = list(events)
    matched = []
    for expected in annotations:
        match = next((index for index, event in enumerate(remaining)
                      if event['action'] == expected['action']
                      and expected['start_ms'] <= event['timestamp_ms'] <= expected['end_ms']), None)
        if match is not None:
            matched.append(remaining.pop(match))
    true_positive = len(matched)
    false_positive = len(remaining)
    false_negative = len(annotations) - true_positive
    return {
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'precision': true_positive / (true_positive + false_positive) if events else None,
        'recall': true_positive / len(annotations) if annotations else None,
        'unmatched_events': remaining,
    }


def score_motion(result, annotations):
    """Report accuracy only when the controller had usable calibrated input."""
    if not result.get('calibration_success') or result.get('usable_ready_frames', 0) == 0:
        return {'eligible': False, 'reason': 'No usable calibrated observations; no accuracy score'}
    return {'eligible': True, 'metrics': summarize_events(result['events'], annotations)}


def replay_clip(clip, path, destination, reference_profile=None):
    video = cv2.VideoCapture(str(path))
    estimator = None
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        if not video.isOpened() or fps <= 0:
            raise ValueError('Unreadable video or invalid FPS: ' + str(path))
        estimator = PoseEstimator()
        recognizer = GestureRecognizer()
        if reference_profile is None:
            recognizer.start_calibration()
        else:
            if not reference_profile.fits_frame():
                raise ValueError('Reference profile has unusable frame geometry')
            # Test harness only: seed a profile already obtained by running the
            # production calibration protocol on the separate reference clip.
            recognizer._apply_profile(reference_profile)
        frames = detected = accepted = unreliable = 0
        ready_frames = 0
        usable_ready_frames = 0
        times = []
        events = []
        actions = Counter()
        output_path = destination / (Path(clip['file']).stem + '.jsonl')
        with output_path.open('w', encoding='utf-8') as stream:
            while True:
                ok, frame = video.read()
                if not ok:
                    break
                # These fixed-rate recordings have a known playback timeline.
                timestamp_ms = round(frames * 1000 / fps)
                frame = cv2.flip(frame, 1)
                started = time.perf_counter()
                result = estimator.process_frame(frame, timestamp_ms)
                times.append((time.perf_counter() - started) * 1000)
                features = estimator.extract_landmarks(result)
                row = {'timestamp_ms': timestamp_ms, 'landmarks': []}
                if result.pose_landmarks:
                    detected += 1
                    points = result.pose_landmarks[0]
                    row['landmarks'] = [[round(p.x, 6), round(p.y, 6), round(p.z, 6),
                                         round(p.visibility, 6), round(p.presence, 6)] for p in points]
                    if min(min(points[i].visibility, points[i].presence)
                           for i in (11, 12, 23, 24)) < 0.5:
                        unreliable += 1
                if features:
                    accepted += 1
                emitted = recognizer.recognize(features, timestamp_ms)
                actions.update(emitted)
                ready_frames += recognizer.is_calibrated
                usable_ready_frames += bool(recognizer.is_calibrated and features)
                events.extend({'timestamp_ms': timestamp_ms, 'action': action} for action in emitted)
                row['actions'] = emitted
                row['features'] = features
                row['calibrated'] = recognizer.is_calibrated
                row['controller_status'] = recognizer.status_message
                stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                frames += 1
        return {
            'file': clip['file'], 'class': clip['class'], 'group': clip['group'],
            'sha256': clip['sha256'], 'fps': fps, 'frames': frames,
            'detected_frames': detected, 'accepted_feature_frames': accepted,
            'unreliable_essential_frames': unreliable, 'actions': dict(actions),
            'events': events,
            'calibration_success': recognizer.is_calibrated,
            'ready_frames': ready_frames,
            'usable_ready_frames': usable_ready_frames,
            'reference_mode': 'production_reference_profile' if reference_profile is not None else 'automatic_calibration',
            'calibration_profile': asdict(recognizer.calibration_profile) if recognizer.calibration_profile else None,
            'final_controller_status': recognizer.status_message,
            'inference_median_ms': round(statistics.median(times), 2) if times else None,
        }
    finally:
        video.release()
        if estimator is not None:
            estimator.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--data', type=Path, default=Path('evaluation/data'))
    parser.add_argument('--output', type=Path, default=Path('evaluation/runs/latest'))
    parser.add_argument('--class-name', action='append', dest='classes')
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    annotations = json.loads(args.annotations.read_text()) if args.annotations else None
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    model = root / 'pose_landmarker_lite.task'
    report = {
        **source_provenance(), 'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
        'dataset': manifest['dataset'], 'conditions': 'mirrored playback; no keyboard injection',
        'annotation_status': 'event windows supplied' if annotations else 'unannotated stress test; no accuracy claim',
        'clips': [],
    }
    for clip in manifest['clips']:
        if args.classes and clip['class'] not in args.classes:
            continue
        path = download_clip(clip, args.data)
        result = replay_clip(clip, path, args.output)
        if annotations is not None and clip['file'] in annotations:
            result['event_score'] = score_motion(result, annotations[clip['file']])
        report['clips'].append(result)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(clip['file'], result['frames'], 'frames', result['actions'], flush=True)


if __name__ == '__main__':
    main()
