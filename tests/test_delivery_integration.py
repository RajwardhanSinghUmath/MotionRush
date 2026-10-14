from camera.clock import now_ns
import contextlib
import io
import unittest
import time
import itertools
from unittest.mock import Mock, patch

import numpy as np

import main
from camera.camera_stream import CaptureRead, CapturedFrame
from control.calibration import CalibrationProfile
from control.gesture_recognizer import GestureRecognizer
from control.keyboard_bridge import KeyboardBridge
from helpers_pose import ImmediatePoseStub
from test_input_scheduler import FakeClock


class MainDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.native = self.stack.enter_context(patch('pydirectinput.SendInput'))
        self.stack.enter_context(patch('pydirectinput.ctypes.windll.user32.GetKeyState', return_value=0))
        self.stack.enter_context(patch('control.keyboard_bridge.time.sleep'))
        self.stack.enter_context(patch('pydirectinput.FAILSAFE', False))
        self.recognizer = GestureRecognizer()
        self.recognizer._apply_profile(CalibrationProfile(.5, .5, .25, .18, .05, .0875, .1375))
        self.clock = FakeClock()
        self.bridge = KeyboardBridge(allow_input=lambda: True, clock=self.clock)
        self.controls = Mock(quit_requested=False, message='Armed: test game')
        self.controls.allow_input.return_value = True
        self.controls.drain_commands.return_value = []
        self.stack.enter_context(patch('main.InputSession', return_value=self.controls))
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.camera = Mock()
        timestamps = itertools.count(4000, 100)
        self.camera.read_frame.side_effect = lambda **kwargs: CaptureRead('frame', CapturedFrame(frame, next(timestamps), now_ns(), 1))
        self.estimator = Mock()
        self.estimator.extract_landmarks.return_value = {'hip_center': (.8, .5)}
        self.overlay = Mock()
        for name in ('draw_skeleton', 'draw_boundary_markers', 'draw_hud'):
            getattr(self.overlay, name).return_value = frame
        for name, instance in (('CameraStream', self.camera), ('PoseWorker', ImmediatePoseStub(self.estimator)),
                               ('GestureRecognizer', self.recognizer), ('KeyboardBridge', self.bridge),
                               ('UIOverlay', self.overlay)):
            self.stack.enter_context(patch('main.' + name, return_value=instance))
        self.stack.enter_context(patch('main.pin_window_always_on_top'))
        def key():
            self.clock.advance_ms(50)
            return -1 if self.clock.ns < 300_000_000 else ord('q')
        self.stack.enter_context(patch.multiple(
            'main.cv2', namedWindow=Mock(), resizeWindow=Mock(), moveWindow=Mock(),
            setWindowProperty=Mock(), imshow=Mock(), waitKey=Mock(side_effect=lambda timeout: key()),
            getWindowProperty=Mock(return_value=1), destroyAllWindows=Mock()))

    def test_partial_double_lane_failure_pauses_main_and_never_retries(self):
        self.recognizer.reset_lane('LEFT')
        self.native.side_effect = [1, 1, 0, 1]
        main.main()
        self.assertEqual(self.recognizer.current_lane, 'CENTER')
        self.assertTrue(self.recognizer.input_fault)
        self.assertEqual(self.native.call_count, 4)
        self.overlay.notify_action.assert_not_called()
        self.camera.release.assert_called_once()
        self.estimator.close.assert_called_once()

    def test_success_updates_lane_and_shows_only_completed_command(self):
        self.native.return_value = 1
        main.main()
        self.assertEqual(self.recognizer.current_lane, 'RIGHT')
        self.assertFalse(self.recognizer.input_fault)
        self.assertEqual(self.native.call_count, 2)
        self.overlay.notify_action.assert_called_once_with('Right')

    def test_exit_retries_failed_release_without_new_key_down(self):
        edges = []
        results = iter([1, 0, 1])

        def native(count, pointer, size):
            edges.append(bool(pointer.contents.ii.ki.dwFlags & 2))
            return next(results)

        self.native.side_effect = native
        main.main()
        self.assertEqual(edges, [False, True, True])
        self.assertTrue(self.recognizer.input_fault)
        self.assertEqual(self.recognizer.current_lane, 'CENTER')

    def test_reset_cannot_resume_when_release_still_fails(self):
        self.native.side_effect = [1, 0, 0, 1]
        def key(timeout):
            self.clock.advance_ms(50)
            return {50: -1, 100: ord('r'), 150: ord('q')}[self.clock.ns // 1_000_000]
        main.cv2.waitKey.side_effect = key
        main.main()
        self.assertTrue(self.recognizer.input_fault)
        self.assertEqual(self.native.call_count, 4)
        self.assertEqual(self.recognizer.current_lane, 'CENTER')

    def test_default_disarmed_main_never_dispatches_a_detected_gesture(self):
        self.controls.allow_input.return_value = False
        main.main()
        self.native.assert_not_called()
        self.overlay.notify_action.assert_not_called()
        self.assertEqual(self.recognizer.current_lane, 'CENTER')

    def test_tracking_loss_pauses_before_any_native_stroke(self):
        self.estimator.extract_landmarks.return_value = None
        main.main()
        self.controls.pause.assert_called_with('Tracking lost; match lane and press F8')
        self.native.assert_not_called()

    def test_global_quit_stops_before_capture(self):
        self.controls.quit_requested = True
        main.main()
        self.camera.read_frame.assert_not_called()
        self.native.assert_not_called()
        self.controls.close.assert_called_once()

    def test_waiting_capture_still_processes_global_quit_without_inference(self):
        def waiting(**kwargs):
            self.controls.quit_requested = True
            return CaptureRead('waiting')

        self.camera.read_frame.side_effect = waiting
        main.main()
        self.estimator.process_frame.assert_not_called()
        self.native.assert_not_called()
        self.camera.release.assert_called_once()

    def test_frame_expired_during_inference_cannot_create_a_command(self):
        main.cv2.waitKey.side_effect = [ord('q')]
        with patch.object(CapturedFrame, 'age_seconds', side_effect=[0, .4]):
            main.main()
        self.native.assert_not_called()
        self.controls.pause.assert_called_with('Pose source expired; input paused')
        self.assertEqual(self.recognizer.current_lane, 'CENTER')

    def test_stale_capture_is_not_sent_to_model_or_keyboard(self):
        self.camera.read_frame.side_effect = None
        self.camera.read_frame.return_value = CaptureRead('stale', message='Camera stalled')
        main.main()
        self.estimator.process_frame.assert_not_called()
        self.native.assert_not_called()
        self.assertIsNone(self.recognizer.waist_position)

    def test_short_wait_between_frames_does_not_reset_calibration(self):
        features = {'hip_center': (.5, .5), 'shoulder_center': (.5, .25),
                    'torso_height': .25, 'shoulder_width': .18, 'confidence': .9,
                    'knee_angles': [175, 175]}
        self.recognizer.start_calibration()
        self.recognizer.recognize(features, 4300, actions_enabled=False)
        self.estimator.extract_landmarks.return_value = features
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.camera.read_frame.side_effect = [CaptureRead('waiting'),
            CaptureRead('frame', CapturedFrame(frame, 4500, now_ns(), 1))]
        main.cv2.waitKey.side_effect = [-1, ord('q')]
        main.main()
        self.assertEqual(len(self.recognizer.calibration._samples), 2)
        self.native.assert_not_called()

    def test_multi_key_sequence_keeps_consuming_pose_results_during_hold_and_gap(self):
        self.recognizer.reset_lane('LEFT')
        self.estimator.extract_landmarks.return_value = {'hip_center': (.8, .4)}
        self.native.return_value = 1
        def key(timeout):
            self.clock.advance_ms(5)
            return ord('q') if self.clock.ns >= 125_000_000 else -1
        main.cv2.waitKey.side_effect = key
        main.main()
        self.assertGreaterEqual(self.estimator.process_frame.call_count, 25)
        self.assertEqual(self.native.call_count, 6)
        self.assertEqual(self.recognizer.current_lane, 'RIGHT')
        self.overlay.notify_action.assert_called_once_with('Jump, Right, Right')

    def test_quit_during_hold_releases_and_acknowledges_uncertain_stroke(self):
        main.cv2.waitKey.side_effect = [ord('q')]
        self.native.return_value = 1
        main.main()
        self.assertEqual(self.native.call_count, 2)
        self.assertTrue(self.recognizer.input_fault)
        self.assertTrue(self.recognizer.input_uncertain)
        self.assertEqual(self.recognizer.pending_actions, ())
        self.assertEqual(self.recognizer.current_lane, 'CENTER')

    def test_quit_between_strokes_commits_completed_lane_step_only(self):
        self.recognizer.reset_lane('LEFT')
        self.native.return_value = 1
        keys = iter([-1, ord('q')])
        def key(timeout):
            self.clock.advance_ms(50)
            return next(keys)
        main.cv2.waitKey.side_effect = key
        main.main()
        self.assertEqual(self.native.call_count, 2)
        self.assertEqual(self.recognizer.current_lane, 'CENTER')
        self.assertTrue(self.recognizer.input_fault)
        self.assertFalse(self.recognizer.input_uncertain)
        self.assertEqual(self.recognizer.pending_actions, ())

    def test_global_pause_cancels_active_hold_before_next_down(self):
        self.native.return_value = 1
        def key(timeout):
            self.controls.allow_input.return_value = False
            self.clock.advance_ms(5)
            return -1 if self.clock.ns == 5_000_000 else ord('q')
        main.cv2.waitKey.side_effect = key
        main.main()
        self.assertEqual(self.native.call_count, 2)
        self.assertTrue(self.recognizer.input_fault)
        self.assertTrue(self.recognizer.input_uncertain)
        self.assertEqual(self.recognizer.pending_actions, ())

    def test_calibration_cancels_and_acknowledges_before_resetting_recognizer(self):
        self.native.return_value = 1
        main.cv2.waitKey.side_effect = [ord('c'), ord('q')]
        main.main()
        self.assertEqual(self.native.call_count, 2)
        self.assertFalse(self.recognizer.is_calibrated)
        self.assertTrue(self.recognizer.input_fault)
        self.assertEqual(self.recognizer.pending_actions, ())

    def test_exception_with_active_hold_releases_and_cleans_up(self):
        self.native.return_value = 1
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.camera.read_frame.side_effect = [CaptureRead('frame', CapturedFrame(frame, 4000, now_ns(), 1)),
                                             RuntimeError('Frame processing failed during delivery')]
        main.cv2.waitKey.side_effect = [-1]
        with self.assertRaisesRegex(RuntimeError, 'Frame processing failed during delivery'):
            main.main()
        self.assertEqual(self.native.call_count, 2)
        self.assertTrue(self.recognizer.input_uncertain)
        self.assertEqual(self.recognizer.pending_actions, ())
        self.camera.release.assert_called_once()
        self.estimator.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
