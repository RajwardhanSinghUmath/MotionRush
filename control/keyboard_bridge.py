import time
import pydirectinput
from config.settings import (
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
    KEY_HOLD_SECONDS, INTER_KEY_GAP_SECONDS,
)

class KeyboardBridge:
    def __init__(self):
        pydirectinput.FAILSAFE = False

    def send_action(self, action):
        """
        Accepts a single action string ('Jump', 'Squat', 'Left', 'Right')
        or a list of actions (e.g. ['Right', 'Right']).
        """
        actions = action if isinstance(action, list) else [action]
        for index, single_action in enumerate(actions):
            if not self._press_single(single_action):
                return False
            if index < len(actions) - 1:
                time.sleep(INTER_KEY_GAP_SECONDS)
        return True

    def _press_single(self, action):
        key = {
            "Jump": KEY_UP, "Squat": KEY_DOWN,
            "Left": KEY_LEFT, "Right": KEY_RIGHT,
        }.get(action)
        if key is None:
            return False
        print(f"[INPUT] Triggered {key.upper()} ({action})")
        # press(_pause=False) still pauses inside its keyDown/keyUp calls.
        # Control each edge explicitly, always releasing even on failure.
        try:
            downed = pydirectinput.keyDown(key, _pause=False)
            if downed:
                time.sleep(KEY_HOLD_SECONDS)
        finally:
            released = pydirectinput.keyUp(key, _pause=False)
        return bool(downed and released)
