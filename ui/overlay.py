import cv2
import time

class UIOverlay:
    def __init__(self):
        self.prev_time = time.time()
        # Define basic connections (start_idx, end_idx)
        self.connections = [
            (11, 12), # shoulders
            (11, 13), (13, 15), # left arm
            (12, 14), (14, 16), # right arm
            (11, 23), (12, 24), # torso
            (23, 24), # hips
            (23, 25), (25, 27), # left leg
            (24, 26), (26, 28)  # right leg
        ]

    def draw_skeleton(self, frame, results):
        if results and hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            height, width, _ = frame.shape
            
            # Draw connections
            for connection in self.connections:
                start_idx, end_idx = connection
                
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    
                    if start_point.visibility > 0.5 and end_point.visibility > 0.5:
                        start_px = (int(start_point.x * width), int(start_point.y * height))
                        end_px = (int(end_point.x * width), int(end_point.y * height))
                        
                        cv2.line(frame, start_px, end_px, (245, 66, 230), 2)
            
            # Draw points
            for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                if idx < len(landmarks):
                    landmark = landmarks[idx]
                    if landmark.visibility > 0.5:
                        px = (int(landmark.x * width), int(landmark.y * height))
                        cv2.circle(frame, px, 4, (245, 117, 66), -1)
                        
        return frame

    def draw_info(self, frame, action, state="Active"):
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

        # Draw Action
        cv2.putText(frame, f'Action: {action}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Draw State (Calibration or Active)
        cv2.putText(frame, f'State: {state}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        # Draw FPS
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        return frame
