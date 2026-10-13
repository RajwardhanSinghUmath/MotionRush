from camera.clock import now_ns
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from camera.camera_stream import CameraStream, LatestFrameSlot, FAILED, ENDED, STATE, SEQUENCE


class BlockedReadDevice:
    def __init__(self, source, clock_start_ns=None):
        self.started = source

    def read_frame(self):
        self.started.value = 1
        time.sleep(60)

    def release(self):
        pass


class BlockedOpenDevice:
    def __init__(self, source, clock_start_ns=None):
        source.value = 1
        time.sleep(60)


class CaptureWorkerTests(unittest.TestCase):
    def video(self, directory):
        path = Path(directory) / 'source.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 25, (32, 24))
        self.assertTrue(writer.isOpened())
        try:
            for index in range(40):
                frame = np.full((24, 32, 3), index * 4, dtype=np.uint8)
                frame[:, :8, 0] = 15
                frame[:, 24:, 0] = 210
                writer.write(frame)
        finally:
            writer.release()
        return path

    def next_frame(self, camera, deadline=8):
        until = time.monotonic() + deadline
        while time.monotonic() < until:
            result = camera.read_frame()
            if result.status == 'frame':
                return result.packet
            self.assertNotIn(result.status, ('error', 'end'), result.message)
        self.fail('Capture worker did not publish a frame')

    def test_spawned_video_capture_skips_backlog_and_preserves_independent_images(self):
        with tempfile.TemporaryDirectory() as directory:
            camera = CameraStream(str(self.video(directory)), playback=True)
            try:
                first = self.next_frame(camera)
                pixels = first.image.copy()
                time.sleep(.20)
                latest = self.next_frame(camera)
                self.assertGreater(latest.sequence, first.sequence + 1)
                self.assertEqual(latest.skipped_frames, latest.sequence - first.sequence - 1)
                self.assertGreater(latest.timestamp_ms, first.timestamp_ms)
                self.assertLess(latest.age_seconds(), .25)
                np.testing.assert_array_equal(first.image, pixels)
                self.assertGreater(first.image[12, 2, 0], first.image[12, 29, 0])
            finally:
                camera.release()
            self.assertFalse(camera.worker_alive)
            camera.release()
            self.assertEqual(camera.read_frame().status, 'closed')

    def test_spawned_recording_reaches_explicit_end(self):
        with tempfile.TemporaryDirectory() as directory:
            camera = CameraStream(str(self.video(directory)), playback=True)
            try:
                statuses = []
                until = time.monotonic() + 8
                while time.monotonic() < until:
                    result = camera.read_frame()
                    statuses.append(result.status)
                    if result.status in ('error', 'end'):
                        break
                self.assertIn('frame', statuses)
                self.assertEqual(statuses[-1], 'end', result.message)
            finally:
                camera.release()

    def test_spawned_open_failure_is_an_error_not_a_frame(self):
        camera = CameraStream('missing-MotionRush-test-recording.avi', playback=True)
        try:
            until = time.monotonic() + 8
            while time.monotonic() < until:
                result = camera.read_frame()
                if result.status == 'error':
                    break
            self.assertEqual(result.status, 'error')
            self.assertIn('Cannot open camera', result.message)
        finally:
            camera.release()

    def test_blocked_driver_read_does_not_block_poll_or_shutdown(self):
        started = mp.get_context('spawn').RawValue('B', 0)
        camera = CameraStream(started, device_factory=BlockedReadDevice)
        try:
            until = time.monotonic() + 8
            while not started.value and time.monotonic() < until:
                camera.read_frame()
            self.assertTrue(started.value, 'Fault-injection worker did not start its blocked read')
            begin = time.monotonic()
            self.assertEqual(camera.read_frame(timeout=.02).status, 'waiting')
            self.assertLess(time.monotonic() - begin, .2)
            begin = time.monotonic()
            camera.release()
            self.assertLess(time.monotonic() - begin, 2.5)
            self.assertTrue(camera.forced_shutdown)
            self.assertFalse(camera.worker_alive)
        finally:
            camera.release()

    def test_blocked_open_has_startup_timeout_and_can_be_stopped(self):
        started = mp.get_context('spawn').RawValue('B', 0)
        camera = CameraStream(started, startup_timeout=5, device_factory=BlockedOpenDevice)
        try:
            until = time.monotonic() + 8
            while not started.value and time.monotonic() < until:
                camera.read_frame()
            self.assertTrue(started.value, 'Fault worker did not enter blocked open')
            # Advance the startup age after confirming the driver-open fault is active.
            camera._origin_ns = now_ns() - 6_000_000_000
            result = camera.read_frame(timeout=.01)
            self.assertEqual(result.status, 'error')
            self.assertIn('startup timed out', result.message)
        finally:
            camera.release()
        self.assertFalse(camera.worker_alive)

    def local_reader(self):
        camera = CameraStream.__new__(CameraStream)
        camera._slot = LatestFrameSlot(mp.get_context('spawn'))
        camera._origin_ns = now_ns()
        camera._startup_timeout = 5
        camera._last_sequence = 0
        camera._last_receipt_ns = None
        camera._released = False
        camera._terminal_error = ''
        camera._process = Mock()
        camera._process.is_alive.return_value = True
        return camera

    def test_stale_frame_is_consumed_once_and_never_returned_for_inference(self):
        camera = self.local_reader()
        camera._slot.publish(np.zeros((8, 8, 3), dtype=np.uint8), 200, now_ns() - 400_000_000)
        self.assertEqual(camera.read_frame(timeout=0).status, 'stale')
        self.assertEqual(camera._last_sequence, 1)
        self.assertEqual(camera.read_frame(timeout=0).status, 'stale')

    def test_held_publication_lock_cannot_make_parent_poll_indefinite(self):
        camera = self.local_reader()
        camera._slot.lock.acquire()
        try:
            begin = time.monotonic()
            self.assertEqual(camera.read_frame(timeout=.02).status, 'waiting')
            self.assertLess(time.monotonic() - begin, .2)
        finally:
            camera._slot.lock.release()

    def test_terminal_failure_is_prioritized_over_pending_frame(self):
        camera = self.local_reader()
        camera._slot.publish(np.zeros((8, 8, 3), dtype=np.uint8), 20, now_ns())
        camera._slot.set_state(FAILED, 'driver failed')
        result = camera.read_frame(timeout=0)
        self.assertEqual(result.status, 'error')
        self.assertEqual(result.message, 'driver failed')
        self.assertIsNone(result.packet)

    def test_oversized_frame_is_bounded_without_changing_aspect(self):
        camera = self.local_reader()
        camera._slot.publish(np.zeros((1080, 1920, 3), dtype=np.uint8), 20, now_ns())
        packet = camera.read_frame(timeout=0).packet
        self.assertEqual(packet.image.shape, (360, 640, 3))
        self.assertEqual(len(camera._slot.pixels), 640 * 480 * 3)

    def test_worker_death_is_not_treated_as_transient_wait(self):
        camera = self.local_reader()
        camera._process.is_alive.return_value = False
        self.assertEqual(camera.read_frame(timeout=0).status, 'error')


if __name__ == '__main__':
    unittest.main()
