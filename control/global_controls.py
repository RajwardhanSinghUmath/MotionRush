"""Windows global controls owned by a dedicated message-pump thread."""

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class WindowIdentity:
    handle: int
    process_id: int
    thread_id: int
    title: str


class WindowsControls:
    HOTKEYS = {1: ('calibrate', 0x76), 2: ('arm', 0x77),
               3: ('pause', 0x78), 4: ('quit', 0x79)}

    def __init__(self):
        self.api = ctypes.WinDLL('user32', use_last_error=True)
        signatures = {
            'RegisterHotKey': ([wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT], wintypes.BOOL),
            'UnregisterHotKey': ([wintypes.HWND, ctypes.c_int], wintypes.BOOL),
            'PeekMessageW': ([ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT], wintypes.BOOL),
            'GetForegroundWindow': ([], wintypes.HWND),
            'GetWindowThreadProcessId': ([wintypes.HWND, ctypes.POINTER(wintypes.DWORD)], wintypes.DWORD),
            'GetWindowTextW': ([wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = arguments, result
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._error = None
        self._cleanup_failed = False

    def foreground(self):
        handle = self.api.GetForegroundWindow()
        if not handle:
            return None
        process = wintypes.DWORD()
        thread = self.api.GetWindowThreadProcessId(handle, ctypes.byref(process))
        title = ctypes.create_unicode_buffer(1024)
        length = self.api.GetWindowTextW(handle, title, len(title))
        if not thread or not process.value or not length or length >= len(title) - 1 or self.api.GetForegroundWindow() != handle:
            return None
        return WindowIdentity(handle, process.value, thread, title.value)

    @property
    def running(self):
        return bool(self._thread and self._thread.is_alive() and self._ready.is_set() and self._error is None)

    @property
    def native_thread_id(self):
        return self._thread.native_id if self._thread else None

    def start(self, on_command, on_tick, on_error):
        if self._thread is not None:
            raise RuntimeError('Global controls cannot be started twice')
        self._thread = threading.Thread(target=self._run, args=(on_command, on_tick, on_error),
                                        name='MotionRush global controls', daemon=True)
        self._thread.start()
        if not self._ready.wait(3) or self._error is not None:
            self.close()
            raise RuntimeError('Global controls unavailable: ' + str(self._error or 'startup timeout'))

    def _run(self, on_command, on_tick, on_error):
        registered = []
        try:
            for identifier, (_, virtual_key) in self.HOTKEYS.items():
                if not self.api.RegisterHotKey(None, identifier, 0x4000, virtual_key):
                    raise RuntimeError(f'F{virtual_key - 0x6F} is already reserved or registration failed')
                registered.append(identifier)
            self._ready.set()
            message = wintypes.MSG()
            while not self._stop.is_set():
                while not self._stop.is_set() and self.api.PeekMessageW(ctypes.byref(message), None, 0x0312, 0x0312, 1):
                    command = self.HOTKEYS.get(message.wParam)
                    if command:
                        # Capture the arm target at the event, not a later frame.
                        on_command(command[0], self.foreground() if command[0] == 'arm' else None)
                on_tick()
                self._stop.wait(.01)
        except Exception as error:
            self._error = error
            on_error()
        finally:
            self._ready.set()
            for identifier in registered:
                try:
                    if not self.api.UnregisterHotKey(None, identifier):
                        self._cleanup_failed = True
                except Exception:
                    self._cleanup_failed = True

    def close(self):
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(3)
            if self._thread.is_alive():
                raise RuntimeError('Global-control thread did not stop')
        if self._cleanup_failed:
            raise RuntimeError('Global hotkey unregister failed')
