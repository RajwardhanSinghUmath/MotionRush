import unittest

from control.gesture_recognizer import GestureRecognizer


def standing(x=0.5, y=0.5, torso=0.25, width=0.18, knees=(175.0, None)):
    return {'hip_center': (x, y), 'torso_height': torso, 'shoulder_width': width,
            'shoulder_center': (x, y - torso), 'image_aspect_ratio': 4 / 3,
            'confidence': 0.9, 'knee_angles': knees}


def calibrate(recognizer, features=None, start_ms=0):
    recognizer.start_calibration()
    for timestamp in range(start_ms, start_ms + 3001, 100):
        actions = recognizer.recognize(features or standing(), timestamp)
        if actions:
            raise AssertionError('Calibration injected actions')


class CalibrationTests(unittest.TestCase):
    def test_startup_never_emits_commands_without_calibration(self):
        recognizer = GestureRecognizer()
        for timestamp in range(0, 5000, 100):
            self.assertEqual(recognizer.recognize(standing(x=0.1, y=0.2), timestamp), [])
        self.assertFalse(recognizer.is_calibrated)

    def test_one_reading_cannot_complete_calibration(self):
        recognizer = GestureRecognizer()
        recognizer.start_calibration()
        self.assertEqual(recognizer.recognize(standing(), 0), [])
        self.assertFalse(recognizer.is_calibrated)
        self.assertEqual(recognizer.calibration.progress, 0)

    def test_stable_standing_creates_valid_player_relative_boundaries(self):
        recognizer = GestureRecognizer()
        calibrate(recognizer)
        self.assertTrue(recognizer.is_calibrated)
        self.assertEqual(recognizer.recognize(standing(), 3100), [])
        self.assertAlmostEqual(recognizer.y_upper, 0.45)
        self.assertAlmostEqual(recognizer.y_lower, 0.5875)
        self.assertTrue(recognizer.calibration_profile.fits_frame())

    def test_thresholds_scale_with_body_size_and_camera_distance(self):
        close, far = GestureRecognizer(), GestureRecognizer()
        calibrate(close, standing(torso=0.3, width=0.2))
        calibrate(far, standing(torso=0.15, width=0.1))
        self.assertTrue(close.is_calibrated and far.is_calibrated)
        self.assertAlmostEqual(close.jump_gap, far.jump_gap * 2)
        self.assertAlmostEqual(close.squat_gap, far.squat_gap * 2)
        self.assertAlmostEqual(close.lane_half_width, far.lane_half_width * 2)

    def test_crouched_or_unseen_legs_cannot_become_standing_reference(self):
        for knees in ((110.0, 120.0), (175.0, 100.0), (None, None)):
            with self.subTest(knees=knees):
                recognizer = GestureRecognizer()
                calibrate(recognizer, standing(knees=knees))
                self.assertFalse(recognizer.is_calibrated)

    def test_edge_position_is_rejected_instead_of_clipping_neutral_zone(self):
        recognizer = GestureRecognizer()
        calibrate(recognizer, standing(x=0.02, y=0.02))
        self.assertFalse(recognizer.is_calibrated)
        self.assertFalse(recognizer.calibration.progress)

    def test_leaning_or_inverted_torso_cannot_become_standing_reference(self):
        for shoulder in ((0.8, 0.25), (0.5, 0.75)):
            features = standing()
            features['shoulder_center'] = shoulder
            recognizer = GestureRecognizer()
            calibrate(recognizer, features)
            self.assertFalse(recognizer.is_calibrated)

    def test_motion_restarts_the_continuous_stability_interval(self):
        recognizer = GestureRecognizer()
        recognizer.start_calibration()
        for timestamp in range(0, 3001, 100):
            recognizer.recognize(standing(x=0.5 if timestamp < 1500 else 0.65), timestamp)
        self.assertFalse(recognizer.is_calibrated)
        for timestamp in range(3100, 4501, 100):
            recognizer.recognize(standing(x=0.65), timestamp)
        self.assertTrue(recognizer.is_calibrated)

    def test_missing_tracking_and_large_time_gaps_reset_progress(self):
        for interrupted in (None, standing()):
            with self.subTest(interrupted=interrupted):
                recognizer = GestureRecognizer()
                recognizer.start_calibration()
                for timestamp in range(0, 1501, 100):
                    recognizer.recognize(standing(), timestamp)
                recognizer.recognize(interrupted, 2000)
                for timestamp in range(2100, 4001, 100):
                    recognizer.recognize(standing(), timestamp)
                self.assertFalse(recognizer.is_calibrated)

    def test_recalibration_disables_actions_and_preserves_game_lane_state(self):
        recognizer = GestureRecognizer()
        calibrate(recognizer)
        recognizer.reset_lane('LEFT')
        recognizer.start_calibration()
        self.assertFalse(recognizer.is_calibrated)
        self.assertEqual(recognizer.current_lane, 'LEFT')
        self.assertEqual(recognizer.recognize(standing(y=0.2), 4000), [])

    def test_invalid_adjustment_does_not_corrupt_calibrated_boundaries(self):
        recognizer = GestureRecognizer()
        calibrate(recognizer)
        original = (recognizer.y_upper, recognizer.y_lower)
        self.assertFalse(recognizer.nudge_vertical(-0.5))
        self.assertEqual((recognizer.y_upper, recognizer.y_lower), original)

    def test_invalid_clock_is_rejected(self):
        recognizer = GestureRecognizer()
        with self.assertRaises(ValueError):
            recognizer.recognize(standing(), float('nan'))


if __name__ == '__main__':
    unittest.main()
