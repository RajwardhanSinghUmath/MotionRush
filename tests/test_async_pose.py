import contextlib
import io
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

import main
from camera.camera_stream import CapturedFrame, CaptureRead
from camera.clock import now_ns
from pose.worker import PoseWorker, PoseRead, PoseSnapshot
from test_pose_worker import BlockedInference


class AsyncPoseIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.camera = Mock()
        self.worker = Mock()
        self.recognizer = Mock()
        self.recognizer.recognize.return_value = []
        self.keyboard = Mock()
        self.keyboard.release_all.return_value = True
        self.overlay = Mock()
        for name in ('draw_skeleton', 'draw_boundary_markers', 'draw_hud'):
            getattr(self.overlay, name).side_effect = lambda image, *args: image
        self.controls = Mock(quit_requested=False, message='Disarmed')
        self.controls.allow_input.return_value = False
        self.controls.drain_commands.return_value = []
        for name, instance in (('InputSession', self.controls), ('CameraStream', self.camera),
                               ('PoseWorker', self.worker), ('GestureRecognizer', self.recognizer),
                               ('KeyboardBridge', self.keyboard), ('UIOverlay', self.overlay)):
            self.stack.enter_context(patch('main.' + name, return_value=instance))
        self.stack.enter_context(patch('main.pin_window_always_on_top'))
        self.wait_key = Mock(side_effect=[-1, -1, ord('q')])
        self.stack.enter_context(patch.multiple(
            'main.cv2', namedWindow=Mock(), resizeWindow=Mock(), moveWindow=Mock(),
            setWindowProperty=Mock(), imshow=Mock(), waitKey=self.wait_key,
            getWindowProperty=Mock(return_value=1), destroyAllWindows=Mock()))

    def packet(self, sequence, received_ns=None):
        return CapturedFrame(np.full((8, 8, 3), sequence, dtype=np.uint8), sequence * 40,
                             now_ns() if received_ns is None else received_ns, sequence)

    def result(self, packet):
        return PoseRead('result', PoseSnapshot(packet, {'hip_center': [.5, .5]}, [], 10))

    def test_cached_pose_is_processed_once_and_drawn_on_its_own_image(self):
        first = self.packet(1)
        newest = self.packet(4)
        self.camera.read_frame.side_effect = [CaptureRead('frame', self.packet(i)) for i in (2, 3, 4)]
        self.worker.read_result.side_effect = [self.result(first), PoseRead('waiting'), self.result(newest)]
        main.main()
        self.assertEqual(self.recognizer.recognize.call_count, 2)
        self.assertEqual([call.args[1] for call in self.recognizer.recognize.call_args_list], [40, 160])
        self.assertEqual(self.controls.note_frame.call_count, 2)
        self.assertEqual([int(call.args[0][0, 0, 0]) for call in self.overlay.draw_skeleton.call_args_list], [1, 1, 4])
        self.keyboard.send_action.assert_not_called()
        self.worker.close.assert_called_once()

    def test_calibration_ignores_results_captured_before_request(self):
        boundary = now_ns()
        old, new = self.packet(1, boundary - 1_000_000), self.packet(2, boundary + 1)
        self.camera.read_frame.return_value = CaptureRead('waiting')
        self.worker.read_result.side_effect = [self.result(old), self.result(new)]
        self.controls.drain_commands.side_effect = [[('calibrate', 1, None)], []]
        self.wait_key.side_effect = [-1, ord('q')]
        with patch('main.now_ns', return_value=boundary):
            main.main()
        self.recognizer.start_calibration.assert_called_once()
        self.recognizer.recognize.assert_called_once_with({'hip_center': [.5, .5]}, 80, actions_enabled=False)

    def test_new_camera_frames_cannot_refresh_an_expired_pose(self):
        expired = self.packet(1, now_ns() - 400_000_000)
        self.camera.read_frame.side_effect = lambda **kwargs: CaptureRead('frame', self.packet(2))
        self.worker.read_result.side_effect = [self.result(expired), PoseRead('waiting'), PoseRead('waiting')]
        main.main()
        self.controls.note_frame.assert_not_called()
        self.assertTrue(all(call.args == (None,) for call in self.recognizer.recognize.call_args_list))
        self.assertTrue(all(call.args[1] is None for call in self.overlay.draw_skeleton.call_args_list))
        self.keyboard.send_action.assert_not_called()

    def test_model_error_releases_input_and_closes_resources(self):
        self.camera.read_frame.return_value = CaptureRead('waiting')
        self.worker.read_result.return_value = PoseRead('error', message='Pose inference timed out')
        main.main()
        self.controls.pause.assert_called_once_with('Pose model stopped; input paused')
        self.keyboard.release_all.assert_called_once()
        self.wait_key.assert_not_called()
        self.worker.close.assert_called_once()
        self.camera.release.assert_called_once()
        self.controls.close.assert_called_once()

    def test_ui_quit_works_while_owned_native_model_call_is_blocked(self):
        import multiprocessing as mp
        context = mp.get_context('spawn')
        configuration = (0, context.RawValue('q', 0), context.RawValue('q', 0), context.RawValue('q', 0))
        worker = PoseWorker(configuration, estimator_factory=BlockedInference)
        self.addCleanup(worker.close)
        until = time.perf_counter() + 8
        while not worker.ready and time.perf_counter() < until:
            result = worker.read_result()
            self.assertNotEqual(result.status, 'error', result.message)
            time.sleep(.005)
        self.assertTrue(worker.ready)
        self.stack.enter_context(patch('main.PoseWorker', return_value=worker))
        sequence = 0
        def capture(**kwargs):
            nonlocal sequence
            sequence += 1
            return CaptureRead('frame', self.packet(sequence))
        self.camera.read_frame.side_effect = capture
        def key(timeout):
            self.assertEqual(timeout, 5)
            time.sleep(.005)
            return ord('q') if configuration[1].value else -1
        self.wait_key.side_effect = key
        before = time.perf_counter()
        main.main()
        self.assertEqual(configuration[1].value, 1)
        self.assertLess(time.perf_counter() - before, 2.5)
        self.assertTrue(worker.forced_shutdown)
        self.assertFalse(worker.worker_alive)
        self.keyboard.send_action.assert_not_called()
        self.camera.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
