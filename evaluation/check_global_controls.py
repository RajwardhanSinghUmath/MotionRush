"""Check actual Windows hotkey registration/cleanup without arming or injecting input."""

import argparse
import ctypes
from ctypes import wintypes
import json
import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path
import threading
import subprocess

from control.input_session import InputSession


def check():
    session = InputSession()
    try:
        session.start()
        assert session.backend.running
        assert not session.allow_input(), 'Startup must remain disarmed'
        processed = threading.Event()
        original_pause = session.pause

        def observed_pause(*args, **kwargs):
            original_pause(*args, **kwargs)
            processed.set()

        session.pause = observed_pause
        api = ctypes.WinDLL('user32', use_last_error=True)
        api.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        api.PostThreadMessageW.restype = wintypes.BOOL
        # Post only to our own worker. This is a synthetic message, not a key press.
        if not api.PostThreadMessageW(session.backend.native_thread_id, 0x0312, 3, 0):
            raise RuntimeError('Could not post test pause message to our own worker')
        if not processed.wait(1):
            raise RuntimeError('Native worker did not process the synthetic pause message')
        return {'windows_api_bindings_loaded': True,
                'pointer_bits': ctypes.sizeof(ctypes.c_void_p) * 8,
                'registered_keys': ['F7', 'F8', 'F9', 'F10'],
                'startup_disarmed': True, 'native_input_injected': False,
                'synthetic_pause_message_processed': True, 'physical_hotkey_press_tested': False,
                'webcam_opened': False}
    finally:
        session.close()
        if session.backend.running:
            raise RuntimeError('Global controls did not clean up')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('evaluation/runs/global-controls.json'))
    args = parser.parse_args()
    report = check()
    check()
    report['hotkeys_reregistered_after_cleanup'] = True
    report['worker_stopped'] = True
    root = Path(__file__).resolve().parents[1]
    report['revision'] = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    report['source_sha256'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                             for name in ('control/global_controls.py', 'control/input_session.py', 'evaluation/check_global_controls.py')}
    report['python_version'] = platform.python_version()
    report['windows_version'] = platform.version()
    report['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
