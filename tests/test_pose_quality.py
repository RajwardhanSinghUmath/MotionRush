import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from pose.estimator import PoseEstimator


def result_with(points):
    return SimpleNamespace(pose_landmarks=[points] if points else [])


class PoseQualityTests(unittest.TestCase):
    def setUp(self):
        self.estimator = PoseEstimator.__new__(PoseEstimator)
        self.points = [SimpleNamespace(x=0.5, y=0.5, visibility=0.9, presence=0.9)
                       for _ in range(33)]
        self.points[11].x, self.points[11].y = 0.4, 0.3
        self.points[12].x, self.points[12].y = 0.6, 0.3
        self.points[23].x, self.points[23].y = 0.43, 0.6
        self.points[24].x, self.points[24].y = 0.57, 0.6

    def test_reliable_pose_keeps_normalized_features(self):
        features = self.estimator.extract_landmarks(result_with(self.points))
        self.assertEqual(features['hip_center'], (0.5, 0.6))
        self.assertAlmostEqual(features['torso_height'], 0.3)
        self.assertAlmostEqual(features['shoulder_width'], 0.2)
        self.assertEqual(features['confidence'], 0.9)

    def test_each_essential_point_requires_visibility_and_presence(self):
        for index in (11, 12, 23, 24):
            for attribute in ('visibility', 'presence'):
                with self.subTest(index=index, attribute=attribute):
                    point = self.points[index]
                    setattr(point, attribute, 0.01)
                    self.assertIsNone(self.estimator.extract_landmarks(result_with(self.points)))
                    setattr(point, attribute, 0.9)

    def test_nonfinite_or_out_of_frame_essential_coordinates_are_rejected(self):
        for attribute, value in [('x', -0.1), ('y', 1.1), ('x', float('nan')),
                                 ('y', float('inf')), ('visibility', float('nan')),
                                 ('presence', None), ('presence', 1.1)]:
            with self.subTest(attribute=attribute, value=value):
                previous = getattr(self.points[23], attribute)
                setattr(self.points[23], attribute, value)
                self.assertIsNone(self.estimator.extract_landmarks(result_with(self.points)))
                setattr(self.points[23], attribute, previous)

    def test_missing_results_and_short_pose_are_tracking_loss(self):
        for result in (None, result_with([]), result_with(self.points[:12])):
            self.assertIsNone(self.estimator.extract_landmarks(result))

    def test_low_quality_unused_joint_does_not_disable_control(self):
        self.points[15].visibility = 0.01
        self.assertIsNotNone(self.estimator.extract_landmarks(result_with(self.points)))

    def test_threshold_boundary_is_accepted(self):
        self.points[23].visibility = self.points[23].presence = 0.5
        self.assertIsNotNone(self.estimator.extract_landmarks(result_with(self.points)))

    def test_missing_confidence_is_rejected(self):
        del self.points[23].presence
        self.assertIsNone(self.estimator.extract_landmarks(result_with(self.points)))

    def test_knee_angle_accounts_for_image_aspect_ratio(self):
        # At 320x240 these segments have offsets (-48,-48) and (48,-48).
        self.points[23].x, self.points[23].y = 0.35, 0.3
        self.points[25].x, self.points[25].y = 0.5, 0.5
        self.points[27].x, self.points[27].y = 0.65, 0.3
        self.estimator._image_aspect_ratio = 4 / 3
        features = self.estimator.extract_landmarks(result_with(self.points))
        self.assertAlmostEqual(features['knee_angles'][0], 90.0)

    def test_actual_recorded_model_outputs(self):
        fixture = Path(__file__).with_name('fixtures') / 'real_pose_quality.json'
        records = json.loads(fixture.read_text())['samples']
        for record in records:
            with self.subTest(file=record['file'], frame=record['frame_index'], condition=record['condition']):
                points = [SimpleNamespace(x=p[0], y=p[1], z=p[2], visibility=p[3], presence=p[4])
                          for p in record['landmarks']]
                accepted = self.estimator.extract_landmarks(result_with(points)) is not None
                self.assertEqual(accepted, record['gate_accepts'])


if __name__ == '__main__':
    unittest.main()
