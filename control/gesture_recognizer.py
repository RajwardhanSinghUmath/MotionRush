import time
from config.settings import JUMP_THRESHOLD, SQUAT_THRESHOLD, LEFT_THRESHOLD, RIGHT_THRESHOLD, ACTION_COOLDOWN

class GestureRecognizer:
    def __init__(self):
        self.baseline = None
        self.last_action_time = 0
        self.calibration_frames = []
        self.is_calibrated = False

    def calibrate(self, features):
        self.calibration_frames.append(features)
        if len(self.calibration_frames) > 30: # Use ~1 sec for calibration
            # Compute average baseline
            avg_shoulder_y = sum(f['shoulder_center'][1] for f in self.calibration_frames) / len(self.calibration_frames)
            avg_hip_y = sum(f['hip_center'][1] for f in self.calibration_frames) / len(self.calibration_frames)
            avg_shoulder_x = sum(f['shoulder_center'][0] for f in self.calibration_frames) / len(self.calibration_frames)
            
            # Use shoulder width as a normalizing factor for X movement
            avg_shoulder_width = sum(abs(f['left_shoulder'][0] - f['right_shoulder'][0]) for f in self.calibration_frames) / len(self.calibration_frames)

            self.baseline = {
                'shoulder_y': avg_shoulder_y,
                'hip_y': avg_hip_y,
                'shoulder_x': avg_shoulder_x,
                'shoulder_width': avg_shoulder_width
            }
            self.is_calibrated = True
            return True
        return False

    def recognize(self, features):
        if not self.is_calibrated or not features:
            return "None"

        current_time = time.time()
        if current_time - self.last_action_time < ACTION_COOLDOWN:
            return "Cooldown"

        # Note: Y-axis is 0 at top, 1 at bottom.
        
        # Vertical movement (Jump / Squat)
        hip_y_diff = features['hip_center'][1] - self.baseline['hip_y']
        
        if hip_y_diff < -JUMP_THRESHOLD:
            self.last_action_time = current_time
            return "Jump"
        elif hip_y_diff > SQUAT_THRESHOLD:
            self.last_action_time = current_time
            return "Squat"

        # Horizontal movement (Left / Right)
        # Note: Camera is flipped horizontally, so left/right might need tuning based on actual perception
        # X-axis is 0 at left, 1 at right (after flip). If person moves to their left, they move to the right of the image.
        shoulder_x_diff = features['shoulder_center'][0] - self.baseline['shoulder_x']
        
        # Normalize by shoulder width
        normalized_x_diff = shoulder_x_diff / (self.baseline['shoulder_width'] + 1e-5)

        if normalized_x_diff > RIGHT_THRESHOLD:
            self.last_action_time = current_time
            return "Right"
        elif normalized_x_diff < -LEFT_THRESHOLD:
            self.last_action_time = current_time
            return "Left"

        return "None"
