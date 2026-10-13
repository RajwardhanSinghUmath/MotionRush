import multiprocessing as mp
import time
from types import SimpleNamespace
import unittest

import numpy as np

from camera.clock import now_ns
from camera.camera_stream import CapturedFrame
from pose.worker import PoseWorker, PoseResultSlot, PHASE, SINCE_NS


class EchoEstimator:
    def __init__(self, configuration):
        self.delay, self.started, self.calls, self.closed = configuration

    def process_frame(self, image, timestamp_ms):
        marker = int(image[0, 0, 0])
        self.started.value = marker
        self.calls.value += 1
        time.sleep(self.delay)
        point = SimpleNamespace(x=marker / 255, y=.5, z=0, visibility=1, presence=1)
        return SimpleNamespace(pose_landmarks=[[point]], marker=marker, timestamp_ms=timestamp_ms)

    def extract_landmarks(self, result):
        return {'hip_center': (result.marker / 255, .5), 'timestamp_seen': result.timestamp_ms}

    def close(self):
        self.closed.value = 1


class BlockedInit:
    def __init__(self, started):
        started.value = 1
        time.sleep(60)


class BlockedInference(EchoEstimator):
    def process_frame(self, image, timestamp_ms):
        self.started.value = 1
        time.sleep(60)


class BlockedClose(EchoEstimator):
    def close(self):
        self.closed.value = 1
        time.sleep(60)


class FailedInit:
    def __init__(self, configuration):
        raise RuntimeError('controlled model initialization failure')


class HugeResponse(EchoEstimator):
    def extract_landmarks(self, result):
        return {'unbounded': 'x' * 40000}


class PoseWorkerTests(unittest.TestCase):
    def configuration(self, delay=0):
        context = mp.get_context('spawn')
        return (delay, context.RawValue('q', 0), context.RawValue('q', 0), context.RawValue('q', 0))

    def worker(self, factory=EchoEstimator, delay=0, **kwargs):
        configuration = self.configuration(delay)
        worker = PoseWorker(configuration, estimator_factory=factory, **kwargs)
        self.addCleanup(worker.close)
        return worker, configuration

    def ready(self, worker):
        until = time.perf_counter() + 10
        while time.perf_counter() < until:
            result = worker.read_result()
            self.assertNotEqual(result.status, 'error', result.message)
            if worker.ready:
                return
            time.sleep(.005)
        self.fail('Model worker did not become ready')

    def response(self, worker, statuses=('result',)):
        until = time.perf_counter() + 5
        while time.perf_counter() < until:
            result = worker.read_result()
            if result.status in statuses:
                return result
            self.assertNotEqual(result.status, 'error', result.message)
            time.sleep(.005)
        self.fail('Model worker did not return the expected status')

    def packet(self, sequence, marker=10):
        return CapturedFrame(np.full((24, 32, 3), marker, dtype=np.uint8), sequence * 40, now_ns(), sequence)

    def test_source_pixels_features_and_timestamps_travel_together(self):
        worker, configuration = self.worker()
        self.ready(worker)
        packet = self.packet(1, 17)
        self.assertTrue(worker.submit(packet))
        snapshot = self.response(worker).snapshot
        self.assertEqual(snapshot.packet.sequence, packet.sequence)
        self.assertEqual(snapshot.packet.received_ns, packet.received_ns)
        self.assertEqual(snapshot.packet.timestamp_ms, packet.timestamp_ms)
        np.testing.assert_array_equal(snapshot.packet.image, packet.image)
        self.assertAlmostEqual(snapshot.features['hip_center'][0], 17 / 255)
        self.assertEqual(snapshot.features['timestamp_seen'], packet.timestamp_ms)
        self.assertEqual(worker.read_result().status, 'waiting')
        self.assertFalse(worker.submit(packet))
        worker.close()
        self.assertEqual(configuration[3].value, 1)
        self.assertTrue(worker.native_close_completed)
        self.assertFalse(worker.worker_alive)
        self.assertEqual(worker.read_result().status, 'closed')

    def test_busy_model_consumes_latest_request_without_processing_the_queue(self):
        worker, configuration = self.worker(delay=.07)
        self.ready(worker)
        worker.submit(self.packet(1, 10))
        until = time.perf_counter() + 2
        while configuration[1].value != 10 and time.perf_counter() < until:
            time.sleep(.005)
        self.assertEqual(configuration[1].value, 10)
        worker.submit(self.packet(2, 20))
        worker.submit(self.packet(3, 30))
        sequences = []
        until = time.perf_counter() + 3
        while time.perf_counter() < until:
            result = worker.read_result()
            if result.status == 'result':
                sequences.append(result.snapshot.packet.sequence)
                if result.snapshot.packet.sequence == 3:
                    np.testing.assert_array_equal(result.snapshot.packet.image, np.full((24, 32, 3), 30, dtype=np.uint8))
                    self.assertEqual(result.snapshot.features['timestamp_seen'], 120)
                    break
            time.sleep(.005)
        self.assertEqual(sequences[-1], 3)
        self.assertNotIn(2, sequences)
        self.assertEqual(configuration[2].value, 2)

    def test_result_expired_in_native_inference_has_no_usable_snapshot(self):
        worker, configuration = self.worker(delay=.32)
        self.ready(worker)
        worker.submit(self.packet(1))
        result = self.response(worker, ('stale',))
        self.assertIsNone(result.snapshot)
        self.assertEqual(worker.last_result_sequence, 1)
        self.assertTrue(worker.worker_alive)

    def test_blocked_initialization_times_out_and_process_can_be_stopped(self):
        started = mp.get_context('spawn').RawValue('B', 0)
        worker = PoseWorker(started, estimator_factory=BlockedInit)
        self.addCleanup(worker.close)
        until = time.perf_counter() + 8
        while not started.value and time.perf_counter() < until:
            worker.read_result()
            time.sleep(.005)
        self.assertEqual(started.value, 1)
        with worker._response.lock:
            worker._response.health[SINCE_NS] = now_ns() - 11_000_000_000
        result = worker.read_result()
        self.assertEqual(result.status, 'error')
        self.assertIn('initialization timed out', result.message)
        before = time.perf_counter()
        worker.close()
        self.assertLess(time.perf_counter() - before, 2.5)
        self.assertTrue(worker.forced_shutdown)
        self.assertFalse(worker.worker_alive)

    def test_active_blocked_inference_times_out_and_does_not_block_parent(self):
        worker, configuration = self.worker(BlockedInference, inference_timeout=.1)
        self.ready(worker)
        worker.submit(self.packet(1))
        result = self.response(worker, ('error',))
        self.assertEqual(configuration[1].value, 1)
        self.assertIn('inference timed out', result.message)
        self.assertFalse(worker.submit(self.packet(2)))
        before = time.perf_counter()
        worker.close()
        self.assertLess(time.perf_counter() - before, 2.5)
        self.assertTrue(worker.forced_shutdown)
        self.assertFalse(worker.worker_alive)

    def test_blocked_native_close_is_stopped_by_parent(self):
        worker, configuration = self.worker(BlockedClose)
        self.ready(worker)
        before = time.perf_counter()
        worker.close()
        self.assertEqual(configuration[3].value, 1)
        self.assertLess(time.perf_counter() - before, 2.5)
        self.assertTrue(worker.forced_shutdown)
        self.assertFalse(worker.native_close_completed)
        self.assertFalse(worker.worker_alive)

    def test_initialization_failure_and_oversized_result_are_terminal(self):
        worker, configuration = self.worker(FailedInit)
        self.assertIn('controlled model initialization failure', self.response(worker, ('error',)).message)
        worker.close()
        worker, configuration = self.worker(HugeResponse)
        self.ready(worker)
        worker.submit(self.packet(1))
        self.assertIn('bounded payload capacity', self.response(worker, ('error',)).message)

    def test_one_brief_response_lock_collision_is_not_a_fatal_timeout(self):
        worker, configuration = self.worker()
        self.ready(worker)
        worker._started_ns = now_ns() - 60_000_000_000
        worker._response.lock.acquire()
        try:
            before = time.perf_counter()
            self.assertEqual(worker.read_result().status, 'waiting')
            self.assertLess(time.perf_counter() - before, .2)
        finally:
            worker._response.lock.release()
        self.assertEqual(worker.read_result().status, 'waiting')

    def test_native_bundled_model_initializes_and_runs_in_owned_process(self):
        worker = PoseWorker()
        self.addCleanup(worker.close)
        self.ready(worker)
        worker.submit(self.packet(1, 0))
        result = self.response(worker, ('result', 'stale'))
        if result.status == 'result':
            self.assertIsNone(result.snapshot.features)
            self.assertEqual(result.snapshot.landmarks, [])
        worker.close()
        self.assertFalse(worker.worker_alive)


if __name__ == '__main__':
    unittest.main()
