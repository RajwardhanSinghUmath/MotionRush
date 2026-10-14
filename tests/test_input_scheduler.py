import contextlib
import unittest
from unittest.mock import Mock, patch

import pydirectinput

from control.keyboard_bridge import KeyboardBridge


class FakeClock:
    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns

    def advance_ms(self, milliseconds):
        self.ns += round(milliseconds * 1e6)


class InputSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('pydirectinput.FAILSAFE', False))
        self.stack.enter_context(patch('pydirectinput.ctypes.windll.user32.GetKeyState', return_value=0))
        self.native = self.stack.enter_context(patch('pydirectinput.SendInput', return_value=1))
        self.sleep = self.stack.enter_context(patch('control.keyboard_bridge.time.sleep', side_effect=AssertionError('Scheduler slept')))
        self.clock = FakeClock()
        self.allowed = True
        self.bridge = KeyboardBridge(allow_input=lambda: self.allowed, clock=self.clock)
        self.addCleanup(self.bridge.close)

    def test_single_stroke_keeps_minimum_hold_without_blocking(self):
        self.assertIsNone(self.bridge.begin_action('Jump'))
        self.assertTrue(self.bridge.busy)
        self.clock.advance_ms(19)
        self.assertIsNone(self.bridge.poll_action())
        self.assertEqual(self.native.call_count, 1)
        self.clock.advance_ms(1)
        result = self.bridge.poll_action()
        self.assertTrue(result)
        self.assertEqual(result.completed, ('Jump',))
        self.assertFalse(self.bridge.busy)
        self.assertIsNone(self.bridge.poll_action())
        self.sleep.assert_not_called()

    def test_two_strokes_preserve_hold_gap_and_edge_order(self):
        edges = []
        self.native.side_effect = lambda count, pointer, size: edges.append((self.clock(), bool(pointer.contents.ii.ki.dwFlags & 2))) or 1
        self.bridge.begin_action(['Right', 'Right'])
        self.clock.advance_ms(20)
        self.assertIsNone(self.bridge.poll_action())
        self.clock.advance_ms(29)
        self.assertIsNone(self.bridge.poll_action())
        self.assertEqual(len(edges), 2)
        self.clock.advance_ms(1)
        self.assertIsNone(self.bridge.poll_action())
        self.clock.advance_ms(20)
        result = self.bridge.poll_action()
        self.assertTrue(result)
        self.assertEqual(edges, [(0, False), (20_000_000, True), (50_000_000, False), (70_000_000, True)])

    def test_late_poll_never_compresses_gap_or_next_hold(self):
        self.bridge.begin_action(['Right', 'Right'])
        self.clock.advance_ms(200)
        self.assertIsNone(self.bridge.poll_action())
        self.assertIsNone(self.bridge.poll_action())
        self.assertEqual(self.native.call_count, 2)
        self.clock.advance_ms(30)
        self.assertIsNone(self.bridge.poll_action())
        self.assertIsNone(self.bridge.poll_action())
        self.assertEqual(self.native.call_count, 3)
        self.clock.advance_ms(20)
        self.assertTrue(self.bridge.poll_action())

    def test_pause_during_hold_releases_immediately_and_is_uncertain(self):
        self.bridge.begin_action('Jump')
        self.clock.advance_ms(1)
        self.allowed = False
        result = self.bridge.poll_action()
        self.assertFalse(result)
        self.assertTrue(result.uncertain)
        self.assertEqual(result.completed, ())
        self.assertEqual(self.native.call_count, 2)
        self.assertIsNone(self.bridge.cancel_action())
        self.assertIsNone(self.bridge.poll_action())

    def test_pause_during_gap_commits_only_completed_prefix(self):
        self.bridge.begin_action(['Right', 'Right'])
        self.clock.advance_ms(20)
        self.bridge.poll_action()
        self.allowed = False
        result = self.bridge.poll_action()
        self.assertEqual(result.completed, ('Right',))
        self.assertFalse(result.uncertain)
        self.assertFalse(result)
        self.assertEqual(self.native.call_count, 2)

    def test_partial_native_failure_stops_remaining_strokes(self):
        self.native.side_effect = [1, 1, 0, 1]
        self.bridge.begin_action(['Right', 'Right'])
        self.clock.advance_ms(20)
        self.bridge.poll_action()
        self.clock.advance_ms(30)
        result = self.bridge.poll_action()
        self.assertEqual(result.completed, ('Right',))
        self.assertTrue(result.uncertain)
        self.assertEqual(self.native.call_count, 4)

    def test_failed_release_blocks_new_down_until_explicit_retry(self):
        self.native.side_effect = [1, 0, 0, 1, 1, 1]
        self.bridge.begin_action('Jump')
        self.clock.advance_ms(20)
        self.assertFalse(self.bridge.poll_action())
        self.assertFalse(self.bridge.begin_action('Right'))
        self.assertEqual(self.native.call_count, 2)
        self.assertFalse(self.bridge.release_all())
        self.assertTrue(self.bridge.release_all())
        self.assertIsNone(self.bridge.begin_action('Right'))
        self.clock.advance_ms(20)
        self.assertTrue(self.bridge.poll_action())

    def test_busy_scheduler_rejects_queued_or_synchronous_commands(self):
        self.bridge.begin_action('Jump')
        self.assertFalse(self.bridge.begin_action('Right'))
        self.assertFalse(self.bridge.send_action('Left'))
        self.assertEqual(self.native.call_count, 1)
        self.clock.advance_ms(20)
        self.assertEqual(self.bridge.poll_action().requested, ('Jump',))

    def test_release_all_preserves_unacknowledged_cancellation(self):
        self.bridge.begin_action('Jump')
        self.assertTrue(self.bridge.release_all())
        self.assertTrue(self.bridge.busy)
        self.assertFalse(self.bridge.begin_action('Right'))
        result = self.bridge.poll_action()
        self.assertEqual(result.requested, ('Jump',))
        self.assertTrue(result.uncertain)
        self.assertFalse(self.bridge.busy)
        self.assertIsNone(self.bridge.poll_action())

    def test_close_releases_and_keeps_one_terminal_result(self):
        self.bridge.begin_action('Jump')
        self.bridge.close()
        self.assertEqual(self.native.call_count, 2)
        self.assertTrue(self.bridge.poll_action().uncertain)
        self.assertIsNone(self.bridge.poll_action())
        self.assertFalse(self.bridge.begin_action('Right'))

    def test_permission_is_checked_immediately_before_native_edge(self):
        bridge = KeyboardBridge(allow_input=Mock(side_effect=[True, False]), clock=self.clock)
        self.addCleanup(bridge.close)
        result = bridge.begin_action('Jump')
        self.assertFalse(result)
        self.assertFalse(result.uncertain)
        self.native.assert_not_called()

    def test_permission_failure_during_hold_still_releases(self):
        self.bridge.begin_action('Jump')
        self.bridge._allow_input = Mock(side_effect=OSError('Focus check failed'))
        result = self.bridge.poll_action()
        self.assertTrue(result.uncertain)
        self.assertIn('Focus check failed', result.error)
        self.assertEqual(self.native.call_count, 2)

    def test_invalid_complete_sequence_is_rejected_before_any_down(self):
        self.assertFalse(self.bridge.begin_action(['Left', 'Unknown']))
        self.native.assert_not_called()

    def test_native_exception_attempts_release_and_marks_uncertainty(self):
        self.native.side_effect = [OSError('Down failed'), 1]
        result = self.bridge.begin_action('Jump')
        self.assertTrue(result.uncertain)
        self.assertEqual(self.native.call_count, 2)


if __name__ == '__main__':
    unittest.main()
