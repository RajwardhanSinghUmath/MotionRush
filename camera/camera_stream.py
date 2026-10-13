import cv2
import time
from config.settings import CAMERA_WIDTH, CAMERA_HEIGHT

class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self._released = False
        try:
            if not self.cap.isOpened():
                raise RuntimeError(f'Cannot open camera {src}; check connection and camera access')
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        except BaseException:
            self.release()
            raise
        self._clock_start_ns = time.monotonic_ns()
        self._last_timestamp_ms = -1

    def read_frame(self):
        """Return (mirrored frame, receipt timestamp in ms), or None on failure."""
        if self._released:
            return None
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
        if not self._released:
            self.cap.release()
            self._released = True
