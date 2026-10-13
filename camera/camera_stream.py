import cv2
import time
from config.settings import CAMERA_WIDTH, CAMERA_HEIGHT

class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self._clock_start_ns = time.monotonic_ns()
        self._last_timestamp_ms = -1

    def read_frame(self):
        """Return (mirrored frame, receipt timestamp in ms), or None on failure."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Timestamp receipt before image processing. Webcam buffers may mean
        # this differs from the sensor's exposure time.
        elapsed_ms = (time.monotonic_ns() - self._clock_start_ns) // 1_000_000
        # Integer rounding must not give MediaPipe duplicate timestamps.
        timestamp_ms = max(elapsed_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        return frame, timestamp_ms

    def release(self):
        self.cap.release()
