import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock

from control.calibration import CalibrationProfile
from control.delivery import simulated_delivery
from control.gesture_recognizer import GestureRecognizer
from control.global_controls import WindowIdentity
from control.input_session import InputSession


def standing(x=.5, y=.5):
    return {'hip_center': (x, y), 'shoulder_center': (x, y - .25),
            'torso_height': .25, 'shoulder_width': .18, 'confidence': .9,
            'knee_angles': [175, 175], 'image_aspect_ratio': 4 / 3}


class InputSessionTests(unittest.TestCase):
    def setUp(self):
        self.target = WindowIdentity(2 ** 40, os.getpid() + 100, 123, 'Test game')
        self.backend = Mock(running=True)
        self.backend.foreground.return_value = self.target
        self.session = InputSession(self.backend)
        self.recognizer = GestureRecognizer()
        self.recognizer._apply_profile(CalibrationProfile(.5, .5, .25, .18, .05, .0875, .1375))

    def arm(self, features=None, age=.01):
        self.session.command('arm', self.target)
        command, generation, target = self.session.drain_commands()[-1]
        return self.session.arm(generation, target, self.recognizer, features or standing(), age)

    def test_startup_is_disarmed_and_does_not_infer_a_target(self):
        self.assertFalse(self.session.allow_input())
        self.backend.foreground.assert_not_called()

    def test_closed_session_cannot_restart_controls(self):
        self.session.close()
        with self.assertRaises(RuntimeError):
            self.session.start()
        self.backend.start.assert_not_called()

    def test_valid_explicit_arm_cannot_emit_a_start_pulse(self):
        self.assertTrue(self.arm())
        self.assertTrue(self.session.allow_input())
        self.assertEqual(self.recognizer.recognize(standing(), 4000), [])

    def test_focus_loss_requires_rearm_even_after_focus_returns(self):
        self.assertTrue(self.arm())
        self.backend.foreground.return_value = None
        self.assertFalse(self.session.allow_input())
        self.backend.foreground.return_value = self.target
        self.assertFalse(self.session.allow_input())
        self.assertTrue(self.arm())

    def test_same_handle_other_process_or_thread_or_title_is_disarmed(self):
        for target in (WindowIdentity(self.target.handle, self.target.process_id + 1, 123, 'Test game'),
                       WindowIdentity(self.target.handle, self.target.process_id, 124, 'Test game'),
                       WindowIdentity(self.target.handle, self.target.process_id, 123, 'Other tab')):
            with self.subTest(target=target):
                self.backend.foreground.return_value = self.target
                self.assertTrue(self.arm())
                self.backend.foreground.return_value = target
                self.assertFalse(self.session.allow_input())

    def test_pause_invalidates_queued_arm_request(self):
        self.session.command('arm', self.target)
        _, generation, target = self.session.drain_commands()[-1]
        self.session.command('pause')
        self.assertFalse(self.session.arm(generation, target, self.recognizer, standing(), .01))
        self.assertFalse(self.session.allow_input())

    def test_target_captured_at_request_cannot_change_before_processing(self):
        self.session.command('arm', self.target)
        _, generation, target = self.session.drain_commands()[-1]
        self.backend.foreground.return_value = WindowIdentity(34, 45, 56, 'Different application')
        self.assertFalse(self.session.arm(generation, target, self.recognizer, standing(), .01))

    def test_preview_process_cannot_be_armed(self):
        self.target = WindowIdentity(123, os.getpid(), 12, 'MotionRush')
        self.backend.foreground.return_value = self.target
        self.assertFalse(self.arm())

    def test_stale_missing_or_crouched_pose_cannot_arm(self):
        for feature, age in ((standing(), .3), (standing(), float('nan')),
                             ({'hip_center': (.5, .5)}, .01),
                             (dict(standing(), knee_angles=[100, 100]), .01)):
            with self.subTest(feature=feature, age=age):
                self.assertFalse(self.arm(feature, age))

    def test_body_lane_must_match_expected_game_lane(self):
        self.recognizer.reset_lane('RIGHT')
        self.assertFalse(self.arm())
        self.assertTrue(self.arm(standing(x=.8)))
        self.assertEqual(self.recognizer.recognize(standing(x=.8), 4000), [])
        self.assertEqual(self.recognizer.recognize(standing(x=.8), 4100), [])
        self.assertEqual(self.recognizer.current_lane, 'RIGHT')

    def test_uncalibrated_or_faulted_controller_cannot_arm(self):
        self.recognizer.is_calibrated = False
        self.assertFalse(self.arm())
        self.recognizer.is_calibrated = True
        self.recognizer.input_fault = True
        self.assertFalse(self.arm())

    def test_stopped_controls_or_foreground_failure_cannot_keep_permission(self):
        self.assertTrue(self.arm())
        self.backend.running = False
        self.assertFalse(self.session.allow_input())
        self.backend.running = True
        self.backend.foreground.side_effect = OSError('foreground inaccessible')
        self.assertFalse(self.arm())

    def test_quit_and_close_invalidate_arming(self):
        self.assertTrue(self.arm())
        self.session.command('quit')
        self.assertTrue(self.session.quit_requested)
        self.assertFalse(self.session.allow_input())
        self.assertFalse(self.arm())
        self.session.close()
        self.backend.close.assert_called_once()

    def test_calibrate_disarms_immediately_before_main_processes_it(self):
        self.assertTrue(self.arm())
        self.session.command('calibrate')
        self.assertFalse(self.session.allow_input())
        self.assertEqual(self.session.drain_commands()[0][0], 'calibrate')

    def test_disarmed_recognition_still_collects_calibration_without_commands(self):
        self.recognizer.start_calibration()
        for timestamp in range(0, 3001, 100):
            self.assertEqual(self.recognizer.recognize(standing(), timestamp, actions_enabled=False), [])
        self.assertTrue(self.recognizer.is_calibrated)
        self.assertFalse(self.session.allow_input())

    def test_real_recorded_jump_trace_is_quiet_until_arm_then_preserves_two_events(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/recorded_gestures.json').read_text())
        self.recognizer._apply_profile(CalibrationProfile(**fixture['profile']))
        rows = fixture['clips'][0]['rows']
        for row in rows:
            self.assertEqual(self.recognizer.recognize(row['features'], row['timestamp_ms'], actions_enabled=False), [])
        self.assertTrue(self.arm(rows[0]['features']))
        actions = []
        for row in rows:
            emitted = self.recognizer.recognize(row['features'], row['timestamp_ms'], actions_enabled=self.session.allow_input())
            if emitted:
                self.recognizer.acknowledge_delivery(simulated_delivery(emitted))
            actions.extend(emitted)
        self.assertEqual(actions, ['Jump', 'Jump'])
