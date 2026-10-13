"""Measure dispatch blocking time with real sleeps and mocked native input."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import pydirectinput

from control.keyboard_bridge import KeyboardBridge
from config.settings import KEY_HOLD_SECONDS, INTER_KEY_GAP_SECONDS


def measure(samples):
    results = {}
    with patch.object(pydirectinput, 'FAILSAFE', False), \
         patch('pydirectinput.SendInput', return_value=1), \
         patch('pydirectinput.ctypes.windll.user32.GetKeyState', return_value=0), \
         contextlib.redirect_stdout(io.StringIO()):
        bridge = KeyboardBridge(allow_input=lambda: True)
        for name, actions in [('single', ['Jump']), ('double', ['Right', 'Right'])]:
            times = []
            for _ in range(samples):
                started = time.perf_counter()
                if not bridge.send_action(actions):
                    raise RuntimeError('Mocked native dispatch failed')
                times.append((time.perf_counter() - started) * 1000)
            results[name] = {
                'samples_ms': [round(value, 2) for value in times],
                'median_ms': round(statistics.median(times), 2),
                'max_ms': round(max(times), 2),
            }
    return {'native_input_mocked': True, 'timing_sleeps_real': True,
            'key_hold_seconds': KEY_HOLD_SECONDS, 'inter_key_gap_seconds': INTER_KEY_GAP_SECONDS,
            'measurements': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--output', type=Path, default=Path('evaluation/runs/input-timing.json'))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    report = measure(args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
