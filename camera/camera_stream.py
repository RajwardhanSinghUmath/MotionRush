import cv2
from config.settings import CAMERA_WIDTH, CAMERA_HEIGHT

class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        self.cap.release()
