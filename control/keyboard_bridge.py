import time
import pydirectinput
from control.delivery import DeliveryResult
from config.settings import (
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
    KEY_HOLD_SECONDS, INTER_KEY_GAP_SECONDS,
)

class KeyboardBridge:
    def __init__(self):
        pydirectinput.FAILSAFE = False
        self._possibly_held = set()

    def send_action(self, action):
        """
        Accepts a single action string ('Jump', 'Squat', 'Left', 'Right')
        or a list of actions (e.g. ['Right', 'Right']).
        """
        actions = tuple(action) if isinstance(action, (list, tuple)) else (action,)
        keys = {'Jump': KEY_UP, 'Squat': KEY_DOWN, 'Left': KEY_LEFT, 'Right': KEY_RIGHT}
        # Validate the entire sequence before injecting any prefix.
        if any(single not in keys for single in actions):
            return DeliveryResult(actions, (), 'Unknown action')
        if self._possibly_held:
            return DeliveryResult(actions, (), 'Key release required', uncertain=True)
        completed = []
        for index, single_action in enumerate(actions):
            error = self._press_single(keys[single_action])
            if error:
                return DeliveryResult(actions, tuple(completed), error, uncertain=True)
            completed.append(single_action)
            if index < len(actions) - 1:
                try:
                    time.sleep(INTER_KEY_GAP_SECONDS)
                except Exception as error:
                    return DeliveryResult(actions, tuple(completed), str(error))
        return DeliveryResult(actions, tuple(completed))

    def _press_single(self, key):
        # press(_pause=False) still pauses inside its keyDown/keyUp calls.
        # Control each edge explicitly, always releasing even on failure.
        error = ''
        self._possibly_held.add(key)
        try:
            downed = pydirectinput.keyDown(key, _pause=False)
            if downed:
                time.sleep(KEY_HOLD_SECONDS)
            else:
                error = 'Key down was not fully inserted'
        except Exception as exception:
            error = str(exception) or type(exception).__name__
        finally:
            try:
                released = pydirectinput.keyUp(key, _pause=False)
                if released:
                    self._possibly_held.discard(key)
                else:
                    error = 'Key up was not fully inserted'
            except Exception as exception:
                error = str(exception) or type(exception).__name__
        return error

    def release_all(self):
        """Retry only possibly held keys; never send a new key down."""
        for key in tuple(self._possibly_held):
            try:
                if pydirectinput.keyUp(key, _pause=False):
                    self._possibly_held.discard(key)
            except Exception:
                pass
        return not self._possibly_held

    def close(self):
        if not self.release_all():
            print('[INPUT] Key release failed; check the game before resuming.')
