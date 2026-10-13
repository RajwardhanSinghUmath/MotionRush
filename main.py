import cv2
import ctypes
import numpy as np
from contextlib import ExitStack
from camera.camera_stream import CameraStream
from pose.estimator import PoseEstimator
from control.gesture_recognizer import GestureRecognizer
from control.keyboard_bridge import KeyboardBridge
from control.input_session import InputSession
from ui.overlay import UIOverlay
from config.settings import (WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_POS_X, WINDOW_POS_Y,
                             CAMERA_WIDTH, CAMERA_HEIGHT, CAPTURE_MAX_AGE_SECONDS)

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
        controls = InputSession()
        resources.callback(controls.close)
        controls.start()
        camera = CameraStream()
        resources.callback(camera.release)
        pose_estimator = PoseEstimator()
        resources.callback(pose_estimator.close)
        recognizer = GestureRecognizer()
        keyboard = KeyboardBridge(allow_input=controls.allow_input)
        resources.callback(keyboard.close)
        overlay = UIOverlay()
        resources.callback(cv2.destroyAllWindows)
        run_controller(camera, pose_estimator, recognizer, keyboard, overlay, controls)


def run_controller(camera, pose_estimator, recognizer, keyboard, overlay, controls):

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
    print('  4. Focus the game and press F8 to arm; input starts disarmed.')
    print('GLOBAL: F7 calibrate | F8 arm focused game | F9 pause | F10 quit')
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
    last_image = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    last_results = None
    have_frame = False

    while True:
        if controls.quit_requested:
            break
        captured_frame = camera.read_frame()
        if captured_frame.status in ('error', 'end', 'closed'):
            controls.pause('Capture stopped')
            keyboard.release_all()
            print('[CAPTURE] ' + (captured_frame.message or captured_frame.status))
            break
        packet = captured_frame.packet
        features = None
        fresh = captured_frame.status == 'frame' and packet.age_seconds() <= CAPTURE_MAX_AGE_SECONDS
        recognizer.camera_message = ''
        if fresh:
            controls.note_frame(packet.received_ns)
            last_image = packet.image
            last_results = pose_estimator.process_frame(packet.image, packet.timestamp_ms)
            features = pose_estimator.extract_landmarks(last_results)
            have_frame = True
            fresh = packet.age_seconds() <= CAPTURE_MAX_AGE_SECONDS
            if not fresh:
                features = None
                controls.pause('Processing too slow; input paused')
                recognizer.camera_message = 'Frame expired during processing'
        elif captured_frame.status == 'stale':
            controls.pause('Camera stalled; input paused')
            recognizer.camera_message = captured_frame.message
        elif captured_frame.status == 'frame':
            controls.pause('Captured frame expired; input paused')
            recognizer.camera_message = 'Captured frame expired before processing'
        elif not have_frame:
            recognizer.camera_message = 'Waiting for camera'

        for command, generation, target in controls.drain_commands():
            if command == 'calibrate':
                recognizer.start_calibration()
                keyboard.release_all()
            elif command == 'arm':
                if keyboard.release_all():
                    controls.arm(generation, target, recognizer, features, packet.received_ns if fresh else None)
        if controls.quit_requested:
            break
        allowed = controls.allow_input()
        if allowed and fresh and not features:
            controls.pause('Tracking lost; match lane and press F8')
            allowed = False
        if not allowed:
            keyboard.release_all()

        actions = []
        if fresh:
            actions = recognizer.recognize(features, packet.timestamp_ms, actions_enabled=allowed)
        elif captured_frame.status == 'stale' or captured_frame.status == 'frame':
            recognizer.recognize(None, packet.timestamp_ms if packet else None, actions_enabled=False)
        if actions:
            delivery = keyboard.send_action(actions)
            recognizer.acknowledge_delivery(delivery)
            if delivery:
                overlay.notify_action(', '.join(delivery.completed))
            else:
                controls.pause('Input failure; center game, focus preview, press R')
                print(f'[INPUT] Paused after delivery failure: {delivery.error}')

        recognizer.input_armed = controls.allow_input()
        recognizer.control_message = controls.message

        # Render Visuals
        frame = overlay.draw_skeleton(last_image.copy(), last_results if fresh else None)
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
            controls.pause('Calibration started; focus game and F8 when ready')
            keyboard.release_all()
            recognizer.start_calibration()
            print("[SYSTEM] Stand upright and still; keep shoulders, hips and a leg visible.")
        elif key == ord('r') or key == ord('R'):
            controls.pause('Lane reset; match lane, focus game and press F8')
            if keyboard.release_all():
                recognizer.reset_lane("CENTER")
                print('[SYSTEM] Expected lane reset to CENTER; match the actual game before resuming.')
            else:
                print('[INPUT] Reset refused: key release still failing.')
        elif key == ord('['):
            controls.pause('Guides changed; match lane and press F8')
            recognizer.adjust_vertical_gap(-0.01)
            print(f"[SYSTEM] Decreased line gap: {recognizer.vertical_gap*2:.2f}")
        elif key == ord(']'):
            controls.pause('Guides changed; match lane and press F8')
            recognizer.adjust_vertical_gap(0.01)
            print(f"[SYSTEM] Increased line gap: {recognizer.vertical_gap*2:.2f}")
        elif key == ord('w') or key == ord('W'):
            controls.pause('Guides changed; match lane and press F8')
            recognizer.nudge_vertical(-0.01)
            print("[SYSTEM] Nudged horizontal lines UP.")
        elif key == ord('s') or key == ord('S'):
            controls.pause('Guides changed; match lane and press F8')
            recognizer.nudge_vertical(0.01)
            print("[SYSTEM] Nudged horizontal lines DOWN.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('[SYSTEM] Interrupted; controller resources released.')
