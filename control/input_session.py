"""Explicit arming and foreground ownership for native input."""

import os
import queue
import threading
from camera.clock import now_ns

from control.calibration import StandingCalibration
from control.global_controls import WindowsControls
from config.settings import CAPTURE_MAX_AGE_SECONDS


class InputSession:
    def __init__(self, backend=None):
        self.backend = backend if backend is not None else WindowsControls()
        self._lock = threading.RLock()
        self._commands = queue.SimpleQueue()
        self._generation = 0
        self._armed = False
        self._target = None
        self._quit = False
        self._closed = False
        self._last_frame_ns = None
        self._message = 'Input disarmed; calibrate, focus game, press F8'

    def start(self):
        if self._closed:
            raise RuntimeError('Cannot start closed input controls')
        self.backend.start(self.command, self.check_focus, lambda: self.pause('Global controls unavailable'))

    @property
    def message(self):
        with self._lock:
            return self._message

    @property
    def quit_requested(self):
        with self._lock:
            return self._quit

    def pause(self, reason='Paused; match lane, focus game, press F8'):
        with self._lock:
            self._generation += 1
            self._armed = False
            self._message = reason

    def command(self, command, target=None):
        with self._lock:
            self.pause()
            if command == 'quit':
                self._quit = True
            elif command in ('arm', 'calibrate'):
                if command == 'arm':
                    self._message = 'Arming requested; checking standing pose'
                self._commands.put((command, self._generation, target))

    def drain_commands(self):
        commands = []
        while True:
            try:
                commands.append(self._commands.get_nowait())
            except queue.Empty:
                return commands

    def note_frame(self, received_ns):
        """Keep acquisition age available to the independent input guard."""
        with self._lock:
            self._last_frame_ns = received_ns

    def arm(self, generation, target, recognizer, features, received_ns):
        with self._lock:
            if generation != self._generation or self._quit or self._closed:
                return False
            if not self.backend.running:
                self.pause('Global controls unavailable')
                return False
            try:
                foreground = self.backend.foreground()
            except Exception:
                self.pause('Cannot verify game focus; input paused')
                return False
            if (not target or target.process_id == os.getpid() or target != foreground):
                self.pause('Focus the game and press F8')
                return False
            if not recognizer.is_calibrated or recognizer.input_fault:
                self.pause('Calibrate or resolve input failure before F8')
                return False
            if recognizer.pending_actions:
                self.pause('Resolve pending input before F8')
                return False
            age = (now_ns() - received_ns) / 1e9 if isinstance(received_ns, int) else float('inf')
            if (not 0 <= age <= CAPTURE_MAX_AGE_SECONDS or not StandingCalibration._valid_features(features)):
                self.pause('Stand upright with a reliable fresh pose, then F8')
                return False
            x, y = features['hip_center']
            lane = 'LEFT' if x < recognizer.x_left else 'RIGHT' if x > recognizer.x_right else 'CENTER'
            if lane != recognizer.current_lane or not recognizer.y_upper <= y <= recognizer.y_lower:
                self.pause('Match expected lane and stand upright, then F8')
                return False
            self._target = target
            self._last_frame_ns = received_ns
            self._armed = True
            self._message = 'Armed: ' + target.title[:45]
            recognizer.prepare_resume()
            return True

    def check_focus(self):
        with self._lock:
            if self._armed:
                try:
                    age = (now_ns() - self._last_frame_ns) / 1e9
                    matches = (0 <= age <= CAPTURE_MAX_AGE_SECONDS and self.backend.running
                               and self.backend.foreground() == self._target)
                except Exception:
                    matches = False
                if not matches:
                    self.pause('Focus or frame freshness lost; match lane and F8')

    def allow_input(self):
        with self._lock:
            self.check_focus()
            return self._armed and not self._closed and not self._quit

    def close(self):
        with self._lock:
            self._closed = True
            self.pause('Input stopped')
        self.backend.close()
