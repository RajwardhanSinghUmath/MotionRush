import time
import pydirectinput
from camera.clock import now_ns
from control.delivery import DeliveryResult
from config.settings import (
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
    KEY_HOLD_SECONDS, INTER_KEY_GAP_SECONDS,
)

class KeyboardBridge:
    def __init__(self, allow_input=None, clock=now_ns):
        self._previous_failsafe = pydirectinput.FAILSAFE
        pydirectinput.FAILSAFE = False
        self._possibly_held = set()
        self._closed = False
        self._clock = clock
        self._active = None
        self._completed = []
        self._phase = None
        self._deadline_ns = 0
        self._ready_result = None
        # Production construction must supply the explicit session permission.
        self._allow_input = allow_input if allow_input is not None else lambda: False

    @property
    def busy(self):
        return self._active is not None or self._ready_result is not None

    @staticmethod
    def _actions(action):
        return tuple(action) if isinstance(action, (list, tuple)) else (action,)

    @staticmethod
    def _keys():
        return {'Jump': KEY_UP, 'Squat': KEY_DOWN, 'Left': KEY_LEFT, 'Right': KEY_RIGHT}

    def _validate(self, actions):
        if self._closed:
            return DeliveryResult(actions, (), 'Input bridge is closed')
        if self.busy:
            return DeliveryResult(actions, (), 'Input delivery is pending')
        if any(single not in self._keys() for single in actions):
            return DeliveryResult(actions, (), 'Unknown action')
        if self._possibly_held:
            return DeliveryResult(actions, (), 'Key release required', uncertain=True)
        return None

    def begin_action(self, action):
        """Start one sequence without waiting; return only an immediate result.

        There is no command queue. Call poll_action until a DeliveryResult arrives
        and acknowledge it before submitting another sequence.
        """
        actions = self._actions(action)
        invalid = self._validate(actions)
        if invalid is not None:
            return invalid
        if not actions:
            return DeliveryResult((), ())
        self._active, self._completed = actions, []
        return self._start_stroke()

    def _finish(self, error='', uncertain=False):
        result = DeliveryResult(self._active, tuple(self._completed), error, uncertain)
        self._active, self._completed, self._phase = None, [], None
        return result

    def _start_stroke(self):
        key = self._keys()[self._active[len(self._completed)]]
        try:
            if not self._allow_input():
                return self._finish('Target is not armed or focused')
            self._phase = 'hold'
            error = self._begin_stroke(key)
            if error:
                uncertain = key in self._possibly_held
                if uncertain:
                    error = self._end_stroke(key) or error
                return self._finish(error, uncertain)
            # Measure the hold from the actual inserted edge, never an old tick.
            self._deadline_ns = self._clock() + round(KEY_HOLD_SECONDS * 1e9)
            return None
        except Exception as exception:
            return self.cancel_action(str(exception) or type(exception).__name__)

    def poll_action(self):
        """Advance at most one timed edge; never sleep or catch up missed ticks."""
        if self._ready_result is not None:
            result, self._ready_result = self._ready_result, None
            return result
        if self._active is None:
            return None
        try:
            if not self._allow_input():
                return self.cancel_action('Target was disarmed during delivery')
        except Exception as exception:
            return self.cancel_action(str(exception) or type(exception).__name__)
        if self._clock() < self._deadline_ns:
            return None
        if self._phase == 'gap':
            return self._start_stroke()
        action = self._active[len(self._completed)]
        error = self._end_stroke(self._keys()[action])
        if error:
            return self._finish(error, uncertain=True)
        self._completed.append(action)
        if len(self._completed) == len(self._active):
            return self._finish()
        self._phase = 'gap'
        self._deadline_ns = self._clock() + round(INTER_KEY_GAP_SECONDS * 1e9)
        return None

    def cancel_action(self, reason='Input delivery cancelled'):
        """Release an active hold and report its completed prefix exactly once."""
        if self._ready_result is not None:
            result, self._ready_result = self._ready_result, None
            return result
        if self._active is None:
            return None
        uncertain = self._phase == 'hold'
        if uncertain:
            key = self._keys()[self._active[len(self._completed)]]
            if key in self._possibly_held:
                reason = self._end_stroke(key) or reason
        return self._finish(reason, uncertain)

    def send_action(self, action):
        """
        Accepts a single action string ('Jump', 'Squat', 'Left', 'Right')
        or a list of actions (e.g. ['Right', 'Right']).
        """
        actions = self._actions(action)
        keys = self._keys()
        # Validate the entire sequence before injecting any prefix.
        invalid = self._validate(actions)
        if invalid is not None:
            return invalid
        completed = []
        for index, single_action in enumerate(actions):
            if not self._allow_input():
                return DeliveryResult(actions, tuple(completed), 'Target is not armed or focused')
            error = self._press_single(keys[single_action])
            if error:
                return DeliveryResult(actions, tuple(completed), error, uncertain=True)
            completed.append(single_action)
            if index < len(actions) - 1:
                try:
                    time.sleep(INTER_KEY_GAP_SECONDS)
                except Exception as error:
                    return DeliveryResult(actions, tuple(completed), str(error) or type(error).__name__)
        return DeliveryResult(actions, tuple(completed))

    def _press_single(self, key):
        # press(_pause=False) still pauses inside its keyDown/keyUp calls.
        # Control each edge explicitly, always releasing even on failure.
        error = self._begin_stroke(key)
        if key not in self._possibly_held:
            return error
        try:
            if not error:
                time.sleep(KEY_HOLD_SECONDS)
                if not self._allow_input():
                    error = 'Target was disarmed during key hold'
        except Exception as exception:
            error = str(exception) or type(exception).__name__
        finally:
            error = self._end_stroke(key) or error
        return error

    def _begin_stroke(self, key):
        if not self._allow_input():
            return 'Target is not armed or focused'
        self._possibly_held.add(key)
        try:
            downed = pydirectinput.keyDown(key, _pause=False)
            return '' if downed else 'Key down was not fully inserted'
        except Exception as exception:
            return str(exception) or type(exception).__name__
        except BaseException:
            self._end_stroke(key)
            raise

    def _end_stroke(self, key):
        try:
            if pydirectinput.keyUp(key, _pause=False):
                self._possibly_held.discard(key)
                return ''
            return 'Key up was not fully inserted'
        except Exception as exception:
            return str(exception) or type(exception).__name__

    def release_all(self):
        """Retry only possibly held keys; never send a new key down."""
        if self._active is not None:
            self._ready_result = self.cancel_action('Key release requested during delivery')
        for key in tuple(self._possibly_held):
            try:
                if pydirectinput.keyUp(key, _pause=False):
                    self._possibly_held.discard(key)
            except Exception:
                pass
        return not self._possibly_held

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            result = self.cancel_action('Input bridge closed during delivery')
            if result is not None:
                self._ready_result = result
            if not self.release_all():
                print('[INPUT] Key release failed; check the game before resuming.')
        finally:
            pydirectinput.FAILSAFE = self._previous_failsafe
