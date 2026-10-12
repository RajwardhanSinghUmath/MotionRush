import time
import pydirectinput
from config.settings import KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT

class KeyboardBridge:
    def __init__(self):
        pydirectinput.FAILSAFE = False

    def send_action(self, action):
        """
        Accepts a single action string ('Jump', 'Squat', 'Left', 'Right')
        or a list of actions (e.g. ['Right', 'Right']).
        """
        if isinstance(action, list):
            for single_action in action:
                self._press_single(single_action)
                time.sleep(0.05)
        else:
            self._press_single(action)

    def _press_single(self, action):
        if action == "Jump":
            print("[INPUT] Triggered KEY_UP (Jump)")
            pydirectinput.press(KEY_UP)
        elif action == "Squat":
            print("[INPUT] Triggered KEY_DOWN (Duck/Roll)")
            pydirectinput.press(KEY_DOWN)
        elif action == "Left":
            print("[INPUT] Triggered KEY_LEFT (Move Left)")
            pydirectinput.press(KEY_LEFT)
        elif action == "Right":
            print("[INPUT] Triggered KEY_RIGHT (Move Right)")
            pydirectinput.press(KEY_RIGHT)
