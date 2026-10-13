import cv2
import ctypes
from contextlib import ExitStack
from camera.camera_stream import CameraStream
from pose.estimator import PoseEstimator
from control.gesture_recognizer import GestureRecognizer
from control.keyboard_bridge import KeyboardBridge
from ui.overlay import UIOverlay
from config.settings import WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_POS_X, WINDOW_POS_Y

WINDOW_NAME = 'Motion Rush (Floating Pop-up)'

def pin_window_always_on_top(win_name):
    """Pins the window on top of all windows without stealing keyboard focus"""
    try:
        hwnd = ctypes.windll.user32.FindWindowW(None, win_name)
        if hwnd:
            HWND_TOPMOST = -1
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOACTIVATE = 0x0010  # Does not steal focus from the game!
            ctypes.windll.user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                             SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
    except Exception:
        pass

def main():
    with ExitStack() as resources:
        camera = CameraStream()
        resources.callback(camera.release)
        pose_estimator = PoseEstimator()
        resources.callback(pose_estimator.close)
        recognizer = GestureRecognizer()
        keyboard = KeyboardBridge()
        overlay = UIOverlay()
        resources.callback(cv2.destroyAllWindows)
        run_controller(camera, pose_estimator, recognizer, keyboard, overlay)


def run_controller(camera, pose_estimator, recognizer, keyboard, overlay):

    # Create compact floating pop-up window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.moveWindow(WINDOW_NAME, WINDOW_POS_X, WINDOW_POS_Y)
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    print("=" * 65)
    print("      MOTION RUSH - FLOATING POP-UP CONTROLLER")
    print("=" * 65)
    print("TIPS FOR PLAYING:")
    print("  1. The camera window will float on top of all other windows.")
    print("  2. CLICK ON YOUR SUBWAY SURFERS GAME to give it keyboard focus!")
    print("  3. Press C, then stand upright and still for calibration before playing.")
    print("\nLINE POSITIONS:")
    print("  - Jump/Squat/Lane guides are scaled to your standing torso.")
    print("\nHOTKEYS (work when camera is active):")
    print("  - [C]     : Start 3-second standing calibration (input stops)")
    print("  - [ or ]  : Decrease / Increase distance between lines")
    print("  - W or S  : Move horizontal lines Up / Down")
    print("  - [R]     : Reset Lane state to Center")
    print("  - [Q]     : Quit")
    print("=" * 65)

    is_pinned = False

    while True:
        captured_frame = camera.read_frame()
        if captured_frame is None:
            print("[ERROR] Failed to capture camera frame.")
            break
        frame, timestamp_ms = captured_frame

        # Estimate pose
        results = pose_estimator.process_frame(frame, timestamp_ms)
        features = pose_estimator.extract_landmarks(results)

        actions = recognizer.recognize(features, timestamp_ms)
        if actions:
            keyboard.send_action(actions)
            overlay.notify_action(", ".join(actions))

        # Render Visuals
        frame = overlay.draw_skeleton(frame, results)
        frame = overlay.draw_boundary_markers(frame, recognizer)
        frame = overlay.draw_hud(frame, recognizer)

        # Show camera window
        cv2.imshow(WINDOW_NAME, frame)

        # Ensure window is pinned topmost without stealing focus
        if not is_pinned:
            pin_window_always_on_top(WINDOW_NAME)
            is_pinned = True

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("[SYSTEM] Quitting application.")
            break
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            # The native window can disappear between display and this check.
            break
        if key == ord('c') or key == ord('C'):
            recognizer.start_calibration()
            print("[SYSTEM] Stand upright and still; keep shoulders, hips and a leg visible.")
        elif key == ord('r') or key == ord('R'):
            recognizer.reset_lane("CENTER")
            print("[SYSTEM] Lane manually reset to CENTER.")
        elif key == ord('['):
            recognizer.adjust_vertical_gap(-0.01)
            print(f"[SYSTEM] Decreased line gap: {recognizer.vertical_gap*2:.2f}")
        elif key == ord(']'):
            recognizer.adjust_vertical_gap(0.01)
            print(f"[SYSTEM] Increased line gap: {recognizer.vertical_gap*2:.2f}")
        elif key == ord('w') or key == ord('W'):
            recognizer.nudge_vertical(-0.01)
            print("[SYSTEM] Nudged horizontal lines UP.")
        elif key == ord('s') or key == ord('S'):
            recognizer.nudge_vertical(0.01)
            print("[SYSTEM] Nudged horizontal lines DOWN.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('[SYSTEM] Interrupted; controller resources released.')
