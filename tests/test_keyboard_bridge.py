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
        self.bridge = KeyboardBridge(allow_input=lambda: True)

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
        result = self.bridge.send_action('Jump')
        self.assertFalse(result)
        self.assertTrue(result.uncertain)
        self.assertEqual(result.completed, ())
        self.assertEqual(self.native_input.call_count, 2)

    def test_partial_double_change_reports_only_completed_prefix(self):
        self.native_input.side_effect = [1, 1, 0, 1]
        result = self.bridge.send_action(['Right', 'Right'])
        self.assertFalse(result)
        self.assertEqual(result.requested, ('Right', 'Right'))
        self.assertEqual(result.completed, ('Right',))
        self.assertTrue(result.uncertain)

    def test_failed_release_blocks_new_down_until_release_succeeds(self):
        self.native_input.side_effect = [1, 0, 0, 1]
        self.assertFalse(self.bridge.send_action('Right'))
        self.assertFalse(self.bridge.send_action('Left'))
        self.assertEqual(self.native_input.call_count, 2)
        self.assertFalse(self.bridge.release_all())
        self.assertTrue(self.bridge.release_all())
        self.assertEqual(self.native_input.call_count, 4)

    def test_invalid_sequence_is_rejected_before_valid_prefix(self):
        result = self.bridge.send_action(['Left', 'Unknown'])
        self.assertFalse(result)
        self.assertEqual(result.completed, ())
        self.assertFalse(result.uncertain)
        self.native_input.assert_not_called()

    def test_key_down_exception_attempts_release_and_reports_uncertainty(self):
        self.native_input.side_effect = [OSError('native failure'), 1]
        result = self.bridge.send_action('Jump')
        self.assertFalse(result)
        self.assertTrue(result.uncertain)
        self.assertEqual(self.native_input.call_count, 2)

    def test_key_up_exception_can_be_retried_during_close(self):
        self.native_input.side_effect = [1, OSError('native release failure'), 1]
        self.assertFalse(self.bridge.send_action('Jump'))
        self.bridge.close()
        self.assertTrue(self.bridge.release_all())
        self.assertEqual(self.native_input.call_count, 3)

    def test_keyboard_interrupt_releases_and_propagates(self):
        self.sleep.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.bridge.send_action('Jump')
        self.assertEqual(self.native_input.call_count, 2)

    def test_numlock_partial_native_insertion_is_uncertain(self):
        with patch('pydirectinput.ctypes.windll.user32.GetKeyState', return_value=1):
            self.native_input.side_effect = [0, 1, 1, 1]
            result = self.bridge.send_action('Right')
        self.assertFalse(result)
        self.assertTrue(result.uncertain)
        self.assertEqual(result.completed, ())
        self.assertEqual(self.native_input.call_count, 4)

    def test_unknown_action_does_not_inject_input(self):
        self.assertFalse(self.bridge.send_action('Unknown'))
        self.native_input.assert_not_called()
        self.sleep.assert_not_called()

    def test_bridge_requires_explicit_permission_by_default(self):
        self.assertFalse(KeyboardBridge().send_action('Jump'))
        self.native_input.assert_not_called()

    def test_closed_bridge_cannot_inject_and_restores_library_failsafe(self):
        with patch.object(pydirectinput, 'FAILSAFE', True):
            bridge = KeyboardBridge(allow_input=lambda: True)
            self.assertFalse(pydirectinput.FAILSAFE)
            bridge.close()
            self.assertTrue(pydirectinput.FAILSAFE)
            self.assertFalse(bridge.send_action('Jump'))
        self.native_input.assert_not_called()

    def test_permission_is_checked_again_immediately_before_key_down(self):
        permission = Mock(side_effect=[True, False])
        bridge = KeyboardBridge(allow_input=permission)
        self.assertFalse(bridge.send_action('Jump'))
        self.native_input.assert_not_called()

    def test_focus_loss_between_strokes_cannot_send_second_down(self):
        allowed = True

        def sleep(seconds):
            nonlocal allowed
            if seconds == INTER_KEY_GAP_SECONDS:
                allowed = False

        self.sleep.side_effect = sleep
        bridge = KeyboardBridge(allow_input=lambda: allowed)
        result = bridge.send_action(['Right', 'Right'])
        self.assertFalse(result)
        self.assertEqual(result.completed, ('Right',))
        self.assertEqual(self.native_input.call_count, 2)

    def test_pause_during_hold_releases_and_reports_uncertainty(self):
        allowed = True

        def sleep(seconds):
            nonlocal allowed
            allowed = False

        self.sleep.side_effect = sleep
        bridge = KeyboardBridge(allow_input=lambda: allowed)
        result = bridge.send_action('Jump')
        self.assertFalse(result)
        self.assertTrue(result.uncertain)
        self.assertEqual(result.completed, ())
        self.assertEqual(self.native_input.call_count, 2)


if __name__ == '__main__':
    unittest.main()
