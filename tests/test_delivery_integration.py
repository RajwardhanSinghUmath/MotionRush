import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import numpy as np

import main
from control.calibration import CalibrationProfile
from control.gesture_recognizer import GestureRecognizer
from control.keyboard_bridge import KeyboardBridge


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
        self.bridge = KeyboardBridge()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.camera = Mock()
        self.camera.read_frame.side_effect = [(frame, 4000), (frame, 4500)]
        self.estimator = Mock()
        self.estimator.extract_landmarks.return_value = {'hip_center': (.8, .5)}
        self.overlay = Mock()
        for name in ('draw_skeleton', 'draw_boundary_markers', 'draw_hud'):
            getattr(self.overlay, name).return_value = frame
        for name, instance in (('CameraStream', self.camera), ('PoseEstimator', self.estimator),
                               ('GestureRecognizer', self.recognizer), ('KeyboardBridge', self.bridge),
                               ('UIOverlay', self.overlay)):
            self.stack.enter_context(patch('main.' + name, return_value=instance))
        self.stack.enter_context(patch('main.pin_window_always_on_top'))
        self.stack.enter_context(patch.multiple(
            'main.cv2', namedWindow=Mock(), resizeWindow=Mock(), moveWindow=Mock(),
            setWindowProperty=Mock(), imshow=Mock(), waitKey=Mock(side_effect=[-1, ord('q')]),
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
        main.cv2.waitKey.side_effect = [ord('r'), ord('q')]
        main.main()
        self.assertTrue(self.recognizer.input_fault)
        self.assertEqual(self.native.call_count, 4)
        self.assertEqual(self.recognizer.current_lane, 'CENTER')


if __name__ == '__main__':
    unittest.main()
