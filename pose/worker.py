"""Bounded latest-request inference in an owned native-model process."""

from dataclasses import dataclass
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from camera.clock import now_ns
from camera.camera_stream import CapturedFrame, LatestFrameSlot, FAILED
from config.settings import (CAPTURE_MAX_AGE_SECONDS, POSE_STARTUP_SECONDS,
                             POSE_INFERENCE_TIMEOUT_SECONDS, POSE_JOIN_SECONDS,
                             POSE_FORCE_JOIN_SECONDS)

INITIALIZING, IDLE, INFERENCING, CLOSING, STOPPED = range(5)
PHASE, SINCE_NS, ACTIVE_SEQUENCE = range(3)


def create_estimator(model_path):
    # Loading native MediaPipe only inside this process also keeps capture spawn light.
    from pose.estimator import PoseEstimator
    return PoseEstimator(model_path=model_path)


@dataclass(frozen=True)
class PoseSnapshot:
    packet: CapturedFrame
    features: object
    landmarks: list
    inference_ms: float

    @property
    def results(self):
        points = [SimpleNamespace(x=p[0], y=p[1], z=p[2], visibility=p[3], presence=p[4])
                  for p in self.landmarks]
        return SimpleNamespace(pose_landmarks=[points] if points else [])


@dataclass(frozen=True)
class PoseRead:
    status: str
    snapshot: PoseSnapshot = None
    message: str = ''


class PoseResultSlot(LatestFrameSlot):
    def __init__(self, context, origin_ns):
        super().__init__(context)
        self.payload = context.RawArray('B', 32768)
        self.payload_size = context.RawValue('I', 0)
        self.health = context.RawArray('q', 3)
        self.health[SINCE_NS] = origin_ns

    def phase(self, phase, sequence=0):
        with self.lock:
            self.health[SINCE_NS] = now_ns()
            self.health[ACTIVE_SEQUENCE] = sequence
            self.health[PHASE] = phase

    def publish_result(self, packet, features, points, inference_ms):
        encoded = json.dumps({'features': features, 'landmarks': points,
                              'inference_ms': inference_ms}, allow_nan=False, separators=(',', ':')).encode('utf-8')
        if len(encoded) > len(self.payload):
            raise ValueError('Pose response exceeds bounded payload capacity')
        with self.lock:
            self.write_locked(packet.image, packet.timestamp_ms, packet.received_ns, packet.sequence)
            self.payload[:len(encoded)] = encoded
            self.payload_size.value = len(encoded)
            self.health[SINCE_NS] = now_ns()
            self.health[PHASE] = IDLE


def request_snapshot(slot, previous_sequence):
    if not slot.lock.acquire(timeout=.002):
        return None
    try:
        _, sequence, width, height, received_ns, timestamp_ms = slot.metadata
        if sequence <= previous_sequence:
            return None
        image = np.frombuffer(slot.pixels, dtype=np.uint8)[:height * width * 3].reshape(height, width, 3).copy()
        return CapturedFrame(image, timestamp_ms, received_ns, sequence)
    finally:
        slot.lock.release()


def pose_process(request, response, stop, model_path, estimator_factory):
    estimator = None
    previous_sequence = 0
    try:
        estimator = estimator_factory(model_path)
        response.phase(IDLE)
        while not stop.value:
            packet = request_snapshot(request, previous_sequence)
            if packet is None:
                time.sleep(.005)
                continue
            previous_sequence = packet.sequence
            if packet.age_seconds() > CAPTURE_MAX_AGE_SECONDS:
                response.publish_result(packet, None, [], 0)
                continue
            response.phase(INFERENCING, packet.sequence)
            started = now_ns()
            result = estimator.process_frame(packet.image, packet.timestamp_ms)
            features = estimator.extract_landmarks(result)
            points = []
            if getattr(result, 'pose_landmarks', None):
                if len(result.pose_landmarks[0]) > 33:
                    raise ValueError('Unexpected pose landmark count')
                for point in result.pose_landmarks[0]:
                    values = [getattr(point, name, None) for name in ('x', 'y', 'z', 'visibility', 'presence')]
                    points.append([value if value is not None and math.isfinite(value) else None for value in values])
            response.publish_result(packet, features, points, (now_ns() - started) / 1e6)
    except BaseException as error:
        response.set_state(FAILED, str(error) or type(error).__name__)
    finally:
        if estimator is not None:
            response.phase(CLOSING)
            try:
                estimator.close()
                response.phase(STOPPED)
            except Exception as error:
                response.set_state(FAILED, 'Pose close failed: ' + str(error))


class PoseWorker:
    def __init__(self, model_path=None, estimator_factory=create_estimator,
                 startup_timeout=POSE_STARTUP_SECONDS, inference_timeout=POSE_INFERENCE_TIMEOUT_SECONDS):
        if any(not math.isfinite(value) or value <= 0 for value in (startup_timeout, inference_timeout)):
            raise ValueError('Pose timeouts must be positive and finite')
        if estimator_factory is create_estimator:
            model_path = Path(model_path) if model_path is not None else Path(__file__).resolve().parents[1] / 'pose_landmarker_lite.task'
            if not model_path.is_file():
                raise FileNotFoundError('Pose model not found: ' + str(model_path))
            model_path = str(model_path.resolve())
        context = mp.get_context('spawn')
        self._request = LatestFrameSlot(context)
        self._response = PoseResultSlot(context, now_ns())
        self._stop = context.RawValue('B', 0)
        self._started_ns = now_ns()
        self._startup_timeout, self._inference_timeout = startup_timeout, inference_timeout
        self._last_submitted = self._last_result = 0
        self._last_timestamp = -1
        self._error = ''
        self._lock_stalled_ns = None
        self._closed = False
        self._ready = False
        self.forced_shutdown = False
        self.native_close_completed = False
        self._process = context.Process(target=pose_process,
            args=(self._request, self._response, self._stop, model_path, estimator_factory),
            name='MotionRush pose', daemon=True)
        try:
            self._process.start()
        except BaseException:
            self._closed = True
            self._process.close()
            raise
        self.worker_pid = self._process.pid

    @property
    def worker_alive(self):
        return bool(self._process and self._process.is_alive())

    @property
    def submitted_sequence(self):
        return self._last_submitted

    @property
    def last_result_sequence(self):
        return self._last_result

    @property
    def ready(self):
        return self._ready and not self._closed and not self._error

    def submit(self, packet):
        if self._closed or self._error or packet.sequence <= self._last_submitted:
            return False
        if packet.age_seconds() > CAPTURE_MAX_AGE_SECONDS:
            return False
        if packet.timestamp_ms <= self._last_timestamp:
            raise ValueError('Pose requests need strictly increasing source timestamps')
        if not self._request.publish(packet.image, packet.timestamp_ms, packet.received_ns,
                                     sequence=packet.sequence, timeout=.002):
            return False
        self._last_submitted, self._last_timestamp = packet.sequence, packet.timestamp_ms
        return True

    def read_result(self):
        if self._closed:
            return PoseRead('closed')
        if self._error:
            return PoseRead('error', message=self._error)
        locked = self._response.lock.acquire(timeout=.002)
        if locked:
            self._lock_stalled_ns = None
            try:
                state, sequence, width, height, received_ns, timestamp_ms = self._response.metadata
                phase, since_ns, _ = self._response.health
                self._ready = phase in (IDLE, INFERENCING) and state != FAILED
                if state == FAILED:
                    self._error = bytes(self._response.error).split(b'\0', 1)[0].decode('utf-8', errors='replace') or 'Pose worker failed'
                elif phase == INITIALIZING and (now_ns() - since_ns) / 1e9 > self._startup_timeout:
                    self._error = 'Pose initialization timed out'
                elif phase == INFERENCING and (now_ns() - since_ns) / 1e9 > self._inference_timeout:
                    self._error = 'Pose inference timed out'
                elif phase in (CLOSING, STOPPED):
                    self._error = 'Pose worker stopped unexpectedly'
                elif sequence > self._last_result:
                    skipped = sequence - self._last_result - 1
                    self._last_result = sequence
                    if not 0 <= (now_ns() - received_ns) / 1e9 <= CAPTURE_MAX_AGE_SECONDS:
                        return PoseRead('stale', message='Pose source frame expired')
                    data = json.loads(bytes(self._response.payload[:self._response.payload_size.value]).decode('utf-8'))
                    image = np.frombuffer(self._response.pixels, dtype=np.uint8)[:height * width * 3].reshape(height, width, 3).copy()
                    packet = CapturedFrame(image, timestamp_ms, received_ns, sequence, skipped)
                    return PoseRead('result', PoseSnapshot(packet, data['features'], data['landmarks'], data['inference_ms']))
            finally:
                self._response.lock.release()
        if not self.worker_alive and not self._error:
            self._error = 'Pose worker exited unexpectedly'
        if not locked:
            if self._lock_stalled_ns is None:
                self._lock_stalled_ns = now_ns()
            elif (now_ns() - self._lock_stalled_ns) / 1e9 > self._inference_timeout:
                self._error = 'Pose response lock unavailable'
        return PoseRead('error', message=self._error) if self._error else PoseRead('waiting')

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.value = 1
        self._process.join(POSE_JOIN_SECONDS)
        if self._response.lock.acquire(timeout=.002):
            try:
                self.native_close_completed = self._response.health[PHASE] == STOPPED
            finally:
                self._response.lock.release()
        if self._process.is_alive():
            self.forced_shutdown = True
            self._process.terminate()
            self._process.join(POSE_FORCE_JOIN_SECONDS)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(POSE_FORCE_JOIN_SECONDS)
        if self._process.is_alive():
            self._closed = False
            raise RuntimeError('Pose worker could not be stopped')
        self._process.close()
        self._process = None
