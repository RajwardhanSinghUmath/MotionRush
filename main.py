import cv2
from camera.camera_stream import CameraStream
from pose.estimator import PoseEstimator
from control.gesture_recognizer import GestureRecognizer
from control.keyboard_bridge import KeyboardBridge
from ui.overlay import UIOverlay

def main():
    camera = CameraStream()
    pose_estimator = PoseEstimator()
    recognizer = GestureRecognizer()
    keyboard = KeyboardBridge()
    overlay = UIOverlay()

    print("Starting Motion Rush...")
    print("Please stand in front of the camera for calibration.")

    while True:
        frame = camera.read_frame()
        if frame is None:
            break

        # Estimate pose
        results = pose_estimator.process_frame(frame)
        
        # Extract features
        features = pose_estimator.extract_landmarks(results)
        
        action = "None"
        state = "Active"

        if features:
            if not recognizer.is_calibrated:
                state = "Calibrating"
                is_done = recognizer.calibrate(features)
                if is_done:
                    print("Calibration complete!")
            else:
                # Recognize gesture
                action = recognizer.recognize(features)
                
                # Send action to game
                if action not in ["None", "Cooldown"]:
                    keyboard.send_action(action)

        # Draw UI
        frame = overlay.draw_skeleton(frame, results)
        frame = overlay.draw_info(frame, action, state)

        # Show frame
        cv2.imshow('Motion Rush Controller', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
