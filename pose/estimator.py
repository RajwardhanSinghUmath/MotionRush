import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseEstimator:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False)
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def process_frame(self, frame):
        # MediaPipe needs RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # We need to provide a monotonically increasing timestamp in VIDEO running mode
        self.timestamp_ms += 33 # assume ~30 fps
        
        results = self.detector.detect_for_video(mp_image, self.timestamp_ms)
        
        return results

    def extract_landmarks(self, results):
        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None
            
        landmarks = results.pose_landmarks[0]
        
        # We need specific points: shoulders and hips
        # Constants from the old mp.solutions.pose.PoseLandmark
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24

        try:
            left_shoulder = landmarks[LEFT_SHOULDER]
            right_shoulder = landmarks[RIGHT_SHOULDER]
            left_hip = landmarks[LEFT_HIP]
            right_hip = landmarks[RIGHT_HIP]
            
            # Calculate centers
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            
            hip_center_x = (left_hip.x + right_hip.x) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2

            shoulder_width = abs(left_shoulder.x - right_shoulder.x)
            torso_height = abs(hip_center_y - shoulder_center_y)
            
            # Return normalized dictionary
            return {
                'shoulder_center': (shoulder_center_x, shoulder_center_y),
                'hip_center': (hip_center_x, hip_center_y),
                'left_shoulder': (left_shoulder.x, left_shoulder.y),
                'right_shoulder': (right_shoulder.x, right_shoulder.y),
                'shoulder_width': shoulder_width,
                'torso_height': torso_height
            }
        except IndexError:
            return None
