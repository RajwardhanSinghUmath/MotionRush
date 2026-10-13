from camera.clock import now_ns
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
import time
from unittest.mock import Mock, patch

import numpy as np

import main
from camera.device_capture import CaptureDevice
from camera.camera_stream import CaptureRead, CapturedFrame
from pose.estimator import PoseEstimator


class ResourceTests(unittest.TestCase):
    def test_camera_open_failure_releases_capture(self):
        capture = Mock()
        capture.isOpened.return_value = False
        with patch('camera.device_capture.cv2.VideoCapture', return_value=capture):
            with self.assertRaisesRegex(RuntimeError, 'Cannot open camera'):
                CaptureDevice()
        capture.release.assert_called_once_with()

    def test_camera_setup_failure_releases_capture(self):
        capture = Mock()
        capture.set.side_effect = RuntimeError('camera setup failed')
        with patch('camera.device_capture.cv2.VideoCapture', return_value=capture):
            with self.assertRaisesRegex(RuntimeError, 'camera setup failed'):
                CaptureDevice()
        capture.release.assert_called_once_with()

    def test_camera_release_is_idempotent(self):
        capture = Mock()
        with patch('camera.device_capture.cv2.VideoCapture', return_value=capture):
            camera = CaptureDevice()
        camera.release()
        camera.release()
        self.assertIsNone(camera.read_frame())
        capture.read.assert_not_called()
        capture.release.assert_called_once_with()

    def test_model_is_resolved_independently_of_launch_directory(self):
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, \
             patch('pose.estimator.vision.PoseLandmarker.create_from_options') as factory:
            try:
                os.chdir(directory)
                estimator = PoseEstimator()
                supplied_path = Path(factory.call_args.args[0].base_options.model_asset_path)
                self.assertTrue(supplied_path.is_absolute())
                self.assertTrue(supplied_path.is_file())
                self.assertEqual(supplied_path.name, 'pose_landmarker_lite.task')
                estimator.close()
            finally:
                os.chdir(previous_directory)

    def test_missing_model_fails_before_detector_allocation(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('pose.estimator.vision.PoseLandmarker.create_from_options') as factory:
            with self.assertRaisesRegex(FileNotFoundError, 'Pose model not found'):
                PoseEstimator(Path(directory) / 'missing.task')
            factory.assert_not_called()

    def test_detector_close_is_idempotent(self):
        with patch('pose.estimator.vision.PoseLandmarker.create_from_options') as factory:
            estimator = PoseEstimator()
            estimator.close()
            estimator.close()
        factory.return_value.close.assert_called_once_with()


class MainLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.patches = contextlib.ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.controls = Mock(quit_requested=False)
        self.controls.allow_input.return_value = False
        self.controls.drain_commands.return_value = []
        self.patches.enter_context(patch('main.InputSession', return_value=self.controls))
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.camera = Mock()
        self.camera.read_frame.side_effect = lambda: CaptureRead('frame', CapturedFrame(self.frame, 28, now_ns(), 1))
        self.estimator = Mock()
        self.estimator.extract_landmarks.return_value = None
        self.camera_factory = self.patches.enter_context(patch('main.CameraStream', return_value=self.camera))
        self.pose_factory = self.patches.enter_context(patch('main.PoseEstimator', return_value=self.estimator))
        recognizer = self.patches.enter_context(patch('main.GestureRecognizer')).return_value
        recognizer.recognize.return_value = []
        self.patches.enter_context(patch('main.KeyboardBridge'))
        overlay = self.patches.enter_context(patch('main.UIOverlay')).return_value
        overlay.draw_skeleton.return_value = self.frame
        overlay.draw_boundary_markers.return_value = self.frame
        overlay.draw_hud.return_value = self.frame
        self.patches.enter_context(patch('main.pin_window_always_on_top'))
        self.window = Mock()
        self.destroy = Mock()
        self.visible = Mock(return_value=1)
        self.patches.enter_context(patch.multiple(
            'main.cv2', namedWindow=self.window, resizeWindow=Mock(), moveWindow=Mock(),
            setWindowProperty=Mock(), imshow=Mock(), waitKey=Mock(return_value=ord('q')),
            getWindowProperty=self.visible, destroyAllWindows=self.destroy,
        ))

    def assert_cleaned_up(self):
        self.controls.close.assert_called_once_with()
        self.camera.release.assert_called_once_with()
        self.estimator.close.assert_called_once_with()
        self.destroy.assert_called_once_with()

    def test_model_initialization_failure_releases_camera(self):
        self.pose_factory.side_effect = RuntimeError('model failed')
        with self.assertRaisesRegex(RuntimeError, 'model failed'):
            main.main()
        self.camera.release.assert_called_once_with()
        self.destroy.assert_not_called()

    def test_window_initialization_failure_releases_all_resources(self):
        self.window.side_effect = RuntimeError('window failed')
        with self.assertRaisesRegex(RuntimeError, 'window failed'):
            main.main()
        self.assert_cleaned_up()

    def test_inference_failure_releases_all_resources(self):
        self.estimator.process_frame.side_effect = RuntimeError('inference failed')
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            main.main()
        self.assert_cleaned_up()

    def test_keyboard_interrupt_releases_all_resources(self):
        self.camera.read_frame.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            main.main()
        self.assert_cleaned_up()

    def test_failed_capture_releases_all_resources(self):
        self.camera.read_frame.side_effect = None
        self.camera.read_frame.return_value = CaptureRead('error', message='Camera failed')
        main.main()
        self.assert_cleaned_up()

    def test_closing_preview_exits_and_releases_all_resources(self):
        main.cv2.waitKey.return_value = -1
        self.visible.return_value = 0
        main.main()
        self.camera.read_frame.assert_called_once_with()
        self.assert_cleaned_up()


if __name__ == '__main__':
    unittest.main()
