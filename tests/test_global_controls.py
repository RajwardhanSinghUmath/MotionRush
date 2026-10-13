import ctypes
from ctypes import wintypes
import os
import queue
import threading
import unittest
from unittest.mock import Mock, patch

from control.calibration import CalibrationProfile
from control.gesture_recognizer import GestureRecognizer
from control.global_controls import WindowsControls
from control.input_session import InputSession


class GlobalControlTests(unittest.TestCase):
    def setUp(self):
        self.api = Mock()
        self.api.RegisterHotKey.return_value = True
        self.api.UnregisterHotKey.return_value = True
        self.messages = queue.Queue()

        def peek(pointer, window, minimum, maximum, remove):
            try:
                identifier = self.messages.get_nowait()
            except queue.Empty:
                return False
            message = ctypes.cast(pointer, ctypes.POINTER(wintypes.MSG)).contents
            message.message = 0x0312
            message.wParam = identifier
            return True

        self.api.PeekMessageW.side_effect = peek
        self.handle = 2 ** 40
        self.api.GetForegroundWindow.return_value = self.handle

        def owner(handle, pointer):
            ctypes.cast(pointer, ctypes.POINTER(wintypes.DWORD)).contents.value = os.getpid() + 100
            return 42

        def title(handle, buffer, length):
            buffer.value = 'Test game'
            return len(buffer.value)

        self.api.GetWindowThreadProcessId.side_effect = owner
        self.api.GetWindowTextW.side_effect = title
        with patch('control.global_controls.ctypes.WinDLL', return_value=self.api):
            self.backend = WindowsControls()
        self.addCleanup(self.backend.close)

    def test_foreground_identity_preserves_64_bit_handle_and_explicit_signatures(self):
        identity = self.backend.foreground()
        self.assertEqual(identity.handle, 2 ** 40)
        self.assertEqual(identity.thread_id, 42)
        self.assertEqual(identity.process_id, os.getpid() + 100)
        self.assertEqual(ctypes.sizeof(self.api.GetForegroundWindow.restype), ctypes.sizeof(ctypes.c_void_p))

    def test_foreground_changing_during_snapshot_is_not_accepted(self):
        self.api.GetForegroundWindow.side_effect = [self.handle, self.handle + 1]
        self.assertIsNone(self.backend.foreground())

    def test_empty_or_truncated_title_cannot_be_used_as_identity(self):
        self.api.GetWindowTextW.side_effect = None
        for length in (0, 1023):
            self.api.GetWindowTextW.return_value = length
            self.assertIsNone(self.backend.foreground())

    def test_partial_registration_failure_unregisters_only_acquired_hotkeys(self):
        self.api.RegisterHotKey.side_effect = [True, True, False]
        with self.assertRaisesRegex(RuntimeError, 'F9'):
            self.backend.start(Mock(), Mock(), Mock())
        self.assertEqual([call.args[1] for call in self.api.UnregisterHotKey.call_args_list], [1, 2])
        self.assertFalse(self.backend.running)

    def test_hotkeys_and_cleanup_are_owned_by_the_same_worker_thread(self):
        register_threads = []
        unregister_threads = []
        self.api.RegisterHotKey.side_effect = lambda *args: register_threads.append(threading.get_ident()) or True
        self.api.UnregisterHotKey.side_effect = lambda *args: unregister_threads.append(threading.get_ident()) or True
        self.backend.start(Mock(), Mock(), Mock())
        self.assertTrue(self.backend.running)
        self.backend.close()
        self.assertEqual(register_threads, unregister_threads)
        self.assertNotEqual(register_threads[0], threading.get_ident())
        self.assertTrue(all(call.args[2] == 0x4000 for call in self.api.RegisterHotKey.call_args_list))

    def test_pause_is_processed_without_main_frame_processing(self):
        session = InputSession(self.backend)
        session.start()
        self.addCleanup(session.close)
        recognizer = GestureRecognizer()
        recognizer._apply_profile(CalibrationProfile(.5, .5, .25, .18, .05, .0875, .1375))
        features = {'hip_center': (.5, .5), 'shoulder_center': (.5, .25),
                    'torso_height': .25, 'shoulder_width': .18, 'confidence': .9,
                    'knee_angles': [175, 175]}
        session.command('arm', self.backend.foreground())
        _, generation, target = session.drain_commands()[0]
        self.assertTrue(session.arm(generation, target, recognizer, features, .01))
        paused = threading.Event()
        original_pause = session.pause

        def pause(reason='Paused'):
            original_pause(reason)
            paused.set()

        session.pause = pause
        self.messages.put(3)
        self.assertTrue(paused.wait(1), 'Background hotkey loop did not process pause')
        self.assertFalse(session.allow_input())

    def test_worker_exception_revokes_permission_and_unregisters(self):
        failed = threading.Event()
        tick = Mock(side_effect=OSError('native API unavailable'))
        try:
            self.backend.start(Mock(), tick, failed.set)
        except RuntimeError:
            pass
        self.assertTrue(failed.wait(1))
        self.backend.close()
        self.assertFalse(self.backend.running)
        self.assertEqual(self.api.UnregisterHotKey.call_count, 4)


if __name__ == '__main__':
    unittest.main()
