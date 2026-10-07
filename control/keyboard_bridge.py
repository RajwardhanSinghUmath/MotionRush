import pydirectinput
from config.settings import KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT

class KeyboardBridge:
    def __init__(self):
        # Optional: Disable pydirectinput failsafe if it causes issues, but usually fine
        pydirectinput.FAILSAFE = False

    def send_action(self, action):
        if action == "Jump":
            print("Sending KEY_UP")
            pydirectinput.press(KEY_UP)
        elif action == "Squat":
            print("Sending KEY_DOWN")
            pydirectinput.press(KEY_DOWN)
        elif action == "Left":
            print("Sending KEY_LEFT")
            pydirectinput.press(KEY_LEFT)
        elif action == "Right":
            print("Sending KEY_RIGHT")
            pydirectinput.press(KEY_RIGHT)
