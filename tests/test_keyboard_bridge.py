import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import pydirectinput

from control.keyboard_bridge import KeyboardBridge
from config.settings import KEY_HOLD_SECONDS, INTER_KEY_GAP_SECONDS


class KeyboardBridgeTests(unittest.TestCase):
    def setUp(self):
        self.patches = contextlib.ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(pydirectinput, 'FAILSAFE', False))
        self.native_input = self.patches.enter_context(patch(
            'pydirectinput.SendInput', return_value=1,
        ))
        self.patches.enter_context(patch(
            'pydirectinput.ctypes.windll.user32.GetKeyState', return_value=0,
        ))
        self.sleep = self.patches.enter_context(patch('control.keyboard_bridge.time.sleep'))
        self.patches.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.bridge = KeyboardBridge()

    def test_single_action_avoids_nested_library_pauses(self):
        self.assertTrue(self.bridge.send_action(['Jump']))
        self.assertEqual(self.native_input.call_count, 2)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [KEY_HOLD_SECONDS])

    def test_double_lane_change_keeps_two_complete_keystrokes(self):
        edges = []

        def send_input(count, pointer, size):
            edges.append(bool(pointer.contents.ii.ki.dwFlags & pydirectinput.KEYEVENTF_KEYUP))
            return 1

        self.native_input.side_effect = send_input
        self.assertTrue(self.bridge.send_action(['Right', 'Right']))
        self.assertEqual(edges, [False, True, False, True])
        self.assertEqual(
            [c.args[0] for c in self.sleep.call_args_list],
            [KEY_HOLD_SECONDS, INTER_KEY_GAP_SECONDS, KEY_HOLD_SECONDS],
        )

    def test_failure_releases_key_and_stops_remaining_actions(self):
        self.native_input.side_effect = [0, 1]
        self.assertFalse(self.bridge.send_action(['Left', 'Right']))
        self.assertEqual(self.native_input.call_count, 2)
        self.sleep.assert_not_called()

    def test_interrupted_hold_still_releases_key(self):
        self.sleep.side_effect = RuntimeError('interrupted hold')
        with self.assertRaisesRegex(RuntimeError, 'interrupted hold'):
            self.bridge.send_action('Jump')
        self.assertEqual(self.native_input.call_count, 2)

    def test_unknown_action_does_not_inject_input(self):
        self.assertFalse(self.bridge.send_action('Unknown'))
        self.native_input.assert_not_called()
        self.sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
