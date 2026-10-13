"""Process-owned capture with a bounded latest-frame slot and bounded polling."""

from dataclasses import dataclass
import math
import multiprocessing as mp
import time

import cv2
import numpy as np

from camera.device_capture import CaptureDevice
from camera.clock import now_ns
from config.settings import (CAMERA_WIDTH, CAMERA_HEIGHT, CAPTURE_POLL_SECONDS,
                             CAPTURE_MAX_AGE_SECONDS, CAPTURE_STARTUP_SECONDS,
                             CAPTURE_JOIN_SECONDS, CAPTURE_FORCE_JOIN_SECONDS)

STARTING, RUNNING, ENDED, FAILED = range(4)
STATE, SEQUENCE, WIDTH, HEIGHT, RECEIVED_NS, TIMESTAMP_MS = range(6)


@dataclass(frozen=True)
class CapturedFrame:
    image: object
    timestamp_ms: int
    received_ns: int
    sequence: int
    skipped_frames: int = 0

    def age_seconds(self):
        age = (now_ns() - self.received_ns) / 1e9
        return age if age >= 0 else float('inf')


@dataclass(frozen=True)
class CaptureRead:
    status: str
    packet: CapturedFrame = None
    message: str = ''


class LatestFrameSlot:
    def __init__(self, context):
        self.pixels = context.RawArray('B', CAMERA_WIDTH * CAMERA_HEIGHT * 3)
        self.metadata = context.RawArray('q', 6)
        self.error = context.RawArray('B', 512)
        self.lock = context.Lock()

    def publish(self, frame, timestamp_ms, received_ns, sequence=None, timeout=None):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError('Capture must provide an 8-bit BGR image')
        height, width = frame.shape[:2]
        if width > CAMERA_WIDTH or height > CAMERA_HEIGHT:
            ratio = min(CAMERA_WIDTH / width, CAMERA_HEIGHT / height)
            frame = cv2.resize(frame, (max(1, int(width * ratio)), max(1, int(height * ratio))))
            height, width = frame.shape[:2]
        locked = self.lock.acquire() if timeout is None else self.lock.acquire(timeout=timeout)
        if not locked:
            return False
        try:
            self.write_locked(frame, timestamp_ms, received_ns, sequence)
            return True
        finally:
            self.lock.release()

    def write_locked(self, frame, timestamp_ms, received_ns, sequence=None):
        """Publish validated pixels and metadata while the caller holds the lock."""
        height, width = frame.shape[:2]
        np.frombuffer(self.pixels, dtype=np.uint8)[:frame.size] = frame.reshape(-1)
        self.metadata[WIDTH], self.metadata[HEIGHT] = width, height
        self.metadata[RECEIVED_NS], self.metadata[TIMESTAMP_MS] = received_ns, timestamp_ms
        self.metadata[SEQUENCE] = self.metadata[SEQUENCE] + 1 if sequence is None else sequence
        self.metadata[STATE] = RUNNING

    def set_state(self, state, message=''):
        encoded = message.encode('utf-8')[:511]
        with self.lock:
            self.error[:] = encoded + bytes(512 - len(encoded))
            self.metadata[STATE] = state


def capture_worker(source, slot, stop, origin_ns, playback, device_factory):
    device = None
    try:
        device = device_factory(source, clock_start_ns=origin_ns)
        slot.set_state(RUNNING)
        interval = 0
        if playback:
            fps = device.cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError('Playback requires valid recording FPS metadata')
            interval = 1 / fps
        next_at = time.perf_counter()
        while not stop.value:
            while interval and not stop.value and time.perf_counter() < next_at:
                time.sleep(min(.005, max(0, next_at - time.perf_counter())))
            if stop.value:
                break
            result = device.read_frame()
            if result is None:
                slot.set_state(ENDED if playback else FAILED,
                               'Recording ended' if playback else 'Camera read failed')
                return
            frame, timestamp_ms = result
            slot.publish(frame, timestamp_ms, device.received_ns)
            next_at = max(next_at + interval, time.perf_counter())
    except BaseException as error:
        slot.set_state(FAILED, str(error) or type(error).__name__)
    finally:
        if device is not None:
            try:
                device.release()
            except Exception as error:
                slot.set_state(FAILED, 'Capture release failed: ' + str(error))


class CameraStream:
    def __init__(self, src=0, playback=False, startup_timeout=CAPTURE_STARTUP_SECONDS,
                 device_factory=CaptureDevice):
        if not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError('Capture startup timeout must be positive and finite')
        self._context = mp.get_context('spawn')
        self._slot = LatestFrameSlot(self._context)
        self._stop = self._context.RawValue('B', 0)
        self._origin_ns = now_ns()
        self._startup_timeout = startup_timeout
        self._last_sequence = 0
        self._last_receipt_ns = None
        self._released = False
        self.forced_shutdown = False
        self._terminal_error = ''
        self._process = self._context.Process(target=capture_worker,
            args=(src, self._slot, self._stop, self._origin_ns, playback, device_factory),
            name='MotionRush capture', daemon=True)
        try:
            self._process.start()
        except BaseException:
            self._released = True
            self._process.close()
            raise
        self.worker_pid = self._process.pid

    @property
    def worker_alive(self):
        return bool(self._process and self._process.is_alive())

    def read_frame(self, timeout=CAPTURE_POLL_SECONDS):
        """Return a fresh latest packet or an explicit wait/stale/end/error status."""
        if not math.isfinite(timeout) or timeout < 0 or timeout > 1:
            raise ValueError('Capture poll timeout must be finite and between zero and one second')
        if self._released:
            return CaptureRead('closed')
        if self._terminal_error:
            return CaptureRead('error', message=self._terminal_error)
        deadline = time.perf_counter() + timeout
        while True:
            # A killed publisher can leave this lock held. Never acquire it indefinitely.
            locked = self._slot.lock.acquire(timeout=min(.002, max(0, deadline - time.perf_counter())))
            if locked:
                try:
                    state, sequence, width, height, received_ns, timestamp_ms = self._slot.metadata
                    message = bytes(self._slot.error).split(b'\0', 1)[0].decode('utf-8', errors='replace')
                    if state == FAILED:
                        self._terminal_error = message or 'Capture worker failed'
                        return CaptureRead('error', message=self._terminal_error)
                    if sequence > self._last_sequence:
                        skipped = sequence - self._last_sequence - 1
                        self._last_sequence = sequence
                        self._last_receipt_ns = received_ns
                        age = (now_ns() - received_ns) / 1e9
                        if age > CAPTURE_MAX_AGE_SECONDS or age < 0:
                            return CaptureRead('stale', message='Captured frame is too old')
                        image = np.frombuffer(self._slot.pixels, dtype=np.uint8)[:height * width * 3].reshape(height, width, 3).copy()
                        return CaptureRead('frame', CapturedFrame(image, timestamp_ms, received_ns, sequence, skipped))
                    if state == ENDED:
                        return CaptureRead('end', message=message)
                finally:
                    self._slot.lock.release()
            if not self.worker_alive:
                self._terminal_error = 'Capture worker exited unexpectedly'
                return CaptureRead('error', message=self._terminal_error)
            if self._last_receipt_ns is None and (now_ns() - self._origin_ns) / 1e9 > self._startup_timeout:
                self._terminal_error = 'Camera startup timed out'
                return CaptureRead('error', message=self._terminal_error)
            if time.perf_counter() >= deadline:
                if self._last_receipt_ns is not None and (now_ns() - self._last_receipt_ns) / 1e9 > CAPTURE_MAX_AGE_SECONDS:
                    return CaptureRead('stale', message='Camera has not provided a fresh frame')
                return CaptureRead('waiting')
            time.sleep(min(.005, max(0, deadline - time.perf_counter())))

    def release(self):
        if self._released:
            return
        self._released = True
        self._stop.value = 1
        self._process.join(CAPTURE_JOIN_SECONDS)
        if self._process.is_alive():
            self.forced_shutdown = True
            self._process.terminate()
            self._process.join(CAPTURE_FORCE_JOIN_SECONDS)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(CAPTURE_FORCE_JOIN_SECONDS)
        if self._process.is_alive():
            self._released = False
            raise RuntimeError('Capture worker could not be stopped')
        self._process.close()
        self._process = None
