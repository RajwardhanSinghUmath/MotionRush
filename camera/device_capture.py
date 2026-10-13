import cv2
from camera.clock import now_ns
from config.settings import CAMERA_WIDTH, CAMERA_HEIGHT

class CaptureDevice:
    """Synchronous device API owned exclusively by the capture worker."""
    def __init__(self, src=0, clock_start_ns=None):
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
        self._clock_start_ns = now_ns() if clock_start_ns is None else clock_start_ns
        self._last_timestamp_ms = -1
        self.received_ns = None

    def read_frame(self):
        """Return (mirrored frame, receipt timestamp in ms), or None on failure."""
        if self._released:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Timestamp receipt before image processing. Webcam buffers may mean
        # this differs from the sensor's exposure time.
        self.received_ns = now_ns()
        elapsed_ms = (self.received_ns - self._clock_start_ns) // 1_000_000
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
