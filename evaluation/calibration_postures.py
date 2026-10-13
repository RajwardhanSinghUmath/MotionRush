"""Review the calibration posture gate on inspected real squat-video frames."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import mediapipe as mp

from control.calibration import StandingCalibration
from evaluation.download import DEFAULT_MANIFEST, download_clip
from pose.estimator import PoseEstimator


# Labels were inspected in source-video contact sheets. Standing candidates
# are not guarantees of usable framing, accurate leg estimation, or full lockout.
LABELS = {
    'v_BodyWeightSquats_g01_c01.avi': {
        'standing_candidate': (0, 10, 20, 73), 'crouched': (35, 48, 60, 116)},
    'v_BodyWeightSquats_g02_c01.avi': {
        'standing_candidate': (37, 46, 55), 'crouched': (0, 10, 20, 73)},
    'v_BodyWeightSquats_g03_c01.avi': {
        'standing_candidate': (0, 10, 20, 95), 'crouched': (36, 48, 71, 119)},
}


def generate(data_directory):
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding='utf-8-sig'))
    selected = [clip for clip in manifest['clips'] if clip['file'] in LABELS]
    samples = []
    for clip in selected:
        path = download_clip(clip, data_directory)
        capture = cv2.VideoCapture(str(path))
        estimator = None
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not capture.isOpened() or fps <= 0:
                raise ValueError('Unreadable source video')
            estimator = PoseEstimator()
            labels = {index: label for label, indices in LABELS[clip['file']].items() for index in indices}
            index = 0
            while index <= max(labels):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('Missing source frame')
                results = estimator.process_frame(cv2.flip(frame, 1), round(index * 1000 / fps))
                if index in labels:
                    points = results.pose_landmarks[0] if results.pose_landmarks else []
                    features = estimator.extract_landmarks(results)
                    samples.append({
                        'file': clip['file'], 'frame_index': index, 'label': labels[index],
                        'landmarks': [[round(p.x, 6), round(p.y, 6), round(p.z, 6),
                                       round(p.visibility, 6), round(p.presence, 6)] for p in points],
                        'usable_calibration_sample': StandingCalibration._valid_features(features),
                    })
                index += 1
        finally:
            capture.release()
            if estimator is not None:
                estimator.close()
    model = Path(__file__).resolve().parents[1] / 'pose_landmarker_lite.task'
    return {
        'description': 'Selected inspected frames from continuous actual inference; not full live calibration sessions',
        'mediapipe_version': mp.__version__, 'opencv_version': cv2.__version__,
        'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
        'image_aspect_ratio': 4 / 3, 'sources': selected, 'samples': samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('evaluation/data'))
    parser.add_argument('--output', type=Path, default=Path('tests/fixtures/calibration_postures.json'))
    args = parser.parse_args()
    fixture = generate(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, separators=(',', ':')), encoding='utf-8')
    for label in ('standing_candidate', 'crouched'):
        samples = [sample for sample in fixture['samples'] if sample['label'] == label]
        print(label, len(samples), 'usable:', sum(sample['usable_calibration_sample'] for sample in samples))


if __name__ == '__main__':
    main()
