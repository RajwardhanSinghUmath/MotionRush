"""Generate quality fixtures from real recordings and explicit image perturbations."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from evaluation.download import DEFAULT_MANIFEST, download_clip
from pose.estimator import PoseEstimator


CONDITIONS = ('original', 'hidden_lower_body', 'upper_body_crop', 'dark', 'blurred')
FILENAMES = ('v_BodyWeightSquats_g01_c01.avi', 'v_BodyWeightSquats_g03_c01.avi',
             'v_WalkingWithDog_g03_c01.avi')


def transform(frame, condition):
    height, width = frame.shape[:2]
    if condition == 'hidden_lower_body':
        frame[int(height * 0.55):] = 0
    elif condition == 'upper_body_crop':
        frame = cv2.resize(frame[:int(height * 0.55)], (width, height))
    elif condition == 'dark':
        frame = (frame.astype(np.float32) * 0.15).astype(np.uint8)
    elif condition == 'blurred':
        frame = cv2.GaussianBlur(frame, (31, 31), 0)
    return frame


def reference_accepts(points):
    if len(points) < 25:
        return False
    for index in (11, 12, 23, 24):
        x, y, z, visibility, presence = points[index]
        if (not all(math.isfinite(v) for v in (x, y, visibility, presence))
                or not (0 <= x <= 1 and 0 <= y <= 1)
                or not (0.5 <= visibility <= 1 and 0.5 <= presence <= 1)):
            return False
    return True


def generate(data_directory):
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding='utf-8-sig'))
    selected = [clip for clip in manifest['clips'] if clip['file'] in FILENAMES]
    samples = []
    for clip in selected:
        path = download_clip(clip, data_directory)
        for condition in CONDITIONS:
            capture = cv2.VideoCapture(str(path))
            estimator = None
            try:
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not capture.isOpened() or fps <= 0:
                    raise ValueError('Unreadable source video: ' + str(path))
                estimator = PoseEstimator()
                for frame_index in (0, 20, 40, 60, 80):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError('Missing fixture source frame')
                    frame = transform(cv2.flip(frame, 1), condition)
                    result = estimator.process_frame(frame, round(frame_index * 1000 / fps))
                    points = result.pose_landmarks[0] if result.pose_landmarks else []
                    serialized = [[round(p.x, 6), round(p.y, 6), round(p.z, 6),
                                   round(p.visibility, 6), round(p.presence, 6)] for p in points]
                    samples.append({
                        'file': clip['file'], 'frame_index': frame_index, 'condition': condition,
                        'landmarks': serialized, 'baseline_accepts': bool(points),
                        'gate_accepts': reference_accepts(serialized),
                    })
            finally:
                capture.release()
                if estimator is not None:
                    estimator.detector.close()
    model = Path(__file__).resolve().parents[1] / 'pose_landmarker_lite.task'
    return {
        'description': 'Actual model outputs from original UCF101 frames and controlled image perturbations; not independent live-camera data',
        'mediapipe_version': mp.__version__, 'opencv_version': cv2.__version__,
        'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
        'reference': 'All essential shoulder/hip x/y in [0,1], visibility and presence in [0.5,1]',
        'transforms': {'hidden_lower_body': 'Black pixels from 55% height downward',
                       'upper_body_crop': 'Resize top 55% to original dimensions',
                       'dark': 'Multiply pixels by 0.15', 'blurred': 'Gaussian blur 31x31'},
        'sources': selected, 'samples': samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('evaluation/data'))
    parser.add_argument('--output', type=Path, default=Path('tests/fixtures/real_pose_quality.json'))
    args = parser.parse_args()
    fixture = generate(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, separators=(',', ':')), encoding='utf-8')
    print('Samples:', len(fixture['samples']), 'baseline accepts:',
          sum(s['baseline_accepts'] for s in fixture['samples']), 'quality gate accepts:',
          sum(s['gate_accepts'] for s in fixture['samples']))


if __name__ == '__main__':
    main()
