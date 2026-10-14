import contextlib
import io
import unittest
import time
from camera.clock import now_ns
from unittest.mock import Mock, patch

import numpy as np

import main
from camera.device_capture import CaptureDevice
from camera.camera_stream import CaptureRead, CapturedFrame
from pose.estimator import PoseEstimator
from helpers_pose import ImmediatePoseStub


class CaptureTimestampTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.capture = Mock()
        self.capture.read.return_value = (True, self.frame)
        self.now_ns = 1_000_000_000
        self.patches = contextlib.ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch(
            'camera.device_capture.cv2.VideoCapture', return_value=self.capture,
        ))
        self.clock = self.patches.enter_context(patch(
            'camera.device_capture.now_ns',
            side_effect=lambda: self.now_ns,
        ))
        self.mirror = self.patches.enter_context(patch(
            'camera.device_capture.cv2.flip', return_value=self.frame,
        ))
        self.camera = CaptureDevice()

    def read_after(self, milliseconds):
        self.now_ns += int(milliseconds * 1_000_000)
        return self.camera.read_frame()

    def test_variable_capture_intervals(self):
        timestamps = [self.read_after(gap)[1] for gap in (28, 45, 120)]
        self.assertEqual(timestamps, [28, 73, 193])

    def test_same_millisecond_frames_remain_strictly_increasing(self):
        timestamps = [self.read_after(gap)[1] for gap in (0, 0.1, 0.1)]
        self.assertEqual(timestamps, [0, 1, 2])

    def test_failed_capture_does_not_create_a_timestamp(self):
        self.assertEqual(self.read_after(28)[1], 28)
        self.capture.read.return_value = (False, None)
        calls_before_failure = self.clock.call_count
        self.assertIsNone(self.read_after(20))
        self.assertEqual(self.clock.call_count, calls_before_failure)
        self.capture.read.return_value = (True, self.frame)
        self.assertEqual(self.read_after(25)[1], 73)

    def test_timestamp_precedes_processing_and_includes_later_delays(self):
        def slow_mirror(frame, direction):
            self.now_ns += 400_000_000
            return frame

        self.mirror.side_effect = slow_mirror
        first_frame, first_timestamp = self.read_after(28)
        second_frame, second_timestamp = self.read_after(45)
        self.assertIs(first_frame, self.frame)
        self.assertIs(second_frame, self.frame)
        self.assertEqual((first_timestamp, second_timestamp), (28, 473))

    def test_pose_estimation_preserves_capture_timestamps(self):
        estimator = PoseEstimator.__new__(PoseEstimator)
        estimator.detector = Mock()
        for gap in (28, 45, 120):
            frame, timestamp_ms = self.read_after(gap)
            result = estimator.process_frame(frame, timestamp_ms)
            self.assertIs(result, estimator.detector.detect_for_video.return_value)
        self.assertEqual(
            [call.args[1] for call in estimator.detector.detect_for_video.call_args_list],
            [28, 73, 193],
        )


class MainTimestampIntegrationTests(unittest.TestCase):
    def test_main_passes_frame_and_timestamp_to_pose_estimator(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        camera = Mock()
        camera.read_frame.side_effect = lambda **kwargs: CaptureRead('frame', CapturedFrame(frame, 125, now_ns(), 1))
        estimator = Mock()
        estimator.extract_landmarks.return_value = None
        overlay = Mock()
        overlay.draw_skeleton.return_value = frame
        overlay.draw_boundary_markers.return_value = frame
        overlay.draw_hud.return_value = frame
        with contextlib.ExitStack() as stack:
            controls = Mock(quit_requested=False)
            controls.allow_input.return_value = False
            controls.drain_commands.return_value = []
            stack.enter_context(patch('main.InputSession', return_value=controls))
            stack.enter_context(patch('main.CameraStream', return_value=camera))
            stack.enter_context(patch('main.PoseWorker', return_value=ImmediatePoseStub(estimator)))
            recognizer = stack.enter_context(patch('main.GestureRecognizer')).return_value
            recognizer.recognize.return_value = []
            keyboard = stack.enter_context(patch('main.KeyboardBridge'))
            keyboard.return_value.cancel_action.return_value = None
            keyboard.return_value.poll_action.return_value = None
            stack.enter_context(patch('main.UIOverlay', return_value=overlay))
            stack.enter_context(patch('main.pin_window_always_on_top'))
            stack.enter_context(patch.multiple(
                'main.cv2', namedWindow=Mock(), resizeWindow=Mock(),
                moveWindow=Mock(), setWindowProperty=Mock(), imshow=Mock(),
                waitKey=Mock(return_value=ord('q')), destroyAllWindows=Mock(),
            ))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            main.main()
        estimator.process_frame.assert_called_once_with(frame, 125)
        recognizer.recognize.assert_called_once_with(None, 125, actions_enabled=False)
        keyboard.return_value.begin_action.assert_not_called()
        camera.release.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
