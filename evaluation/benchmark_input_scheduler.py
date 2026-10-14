"""Compare input blocking and original-video transport with native input mocked."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
from unittest.mock import patch

from camera.camera_stream import CameraStream
from control.keyboard_bridge import KeyboardBridge
from evaluation.benchmark_pose_worker import wait_ready
from evaluation.download import download_clip
from pose.worker import PoseWorker
from config.settings import KEY_HOLD_SECONDS, INTER_KEY_GAP_SECONDS, CAPTURE_MAX_AGE_SECONDS


def summary(values):
    return {'median_ms': round(statistics.median(values), 3), 'max_ms': round(max(values), 3)} if values else None


def dispatch_measurements(samples):
    report = []
    for name, actions in (('single', ['Jump']), ('double', ['Right', 'Right']),
                           ('triple', ['Jump', 'Right', 'Right'])):
        for mode in ('blocking', 'scheduled'):
            calls, elapsed, holds, gaps = [], [], [], []
            for _ in range(samples):
                edges = []
                def native(count, pointer, size):
                    edges.append((time.perf_counter(), bool(pointer.contents.ii.ki.dwFlags & 2)))
                    return 1
                bridge = KeyboardBridge(allow_input=lambda: True)
                try:
                    with patch('pydirectinput.SendInput', side_effect=native):
                        started = time.perf_counter()
                        if mode == 'blocking':
                            result = bridge.send_action(actions)
                            calls.append((time.perf_counter() - started) * 1000)
                        else:
                            result = bridge.begin_action(actions)
                            calls.append((time.perf_counter() - started) * 1000)
                            while result is None:
                                before = time.perf_counter()
                                result = bridge.poll_action()
                                calls.append((time.perf_counter() - before) * 1000)
                                if result is None:
                                    time.sleep(.005)
                        elapsed.append((time.perf_counter() - started) * 1000)
                        if not result:
                            raise RuntimeError('Mocked dispatch failed: ' + result.error)
                        if [up for _, up in edges] != [False, True] * len(actions):
                            raise RuntimeError('Unexpected native event order')
                        holds.extend((edges[i + 1][0] - edges[i][0]) * 1000 for i in range(0, len(edges), 2))
                        gaps.extend((edges[i + 1][0] - edges[i][0]) * 1000 for i in range(1, len(edges) - 1, 2))
                finally:
                    bridge.close()
            report.append({'sequence': name, 'mode': mode, 'samples': samples,
                           'individual_call_blocking': summary(calls), 'full_delivery_elapsed': summary(elapsed),
                           'minimum_observed_hold_ms': round(min(holds), 3),
                           'minimum_observed_gap_ms': round(min(gaps), 3) if gaps else None})
    return report


def recording_measurement(clip, path, mode):
    worker = PoseWorker()
    camera = None
    latest_receipt_ns = None
    permission = lambda: latest_receipt_ns is not None and 0 <= (time.perf_counter_ns() - latest_receipt_ns) / 1e9 <= CAPTURE_MAX_AGE_SECONDS
    bridge = KeyboardBridge(allow_input=permission)
    captured = skipped = expired = completed = 0
    ages, poll_calls, deliveries = [], [], []
    next_action_sequence = 1
    ended = False
    try:
        startup_ms = wait_ready(worker)
        camera = CameraStream(str(path), playback=True)
        deadline = time.perf_counter() + 90
        while time.perf_counter() < deadline:
            before = time.perf_counter()
            if not ended:
                read = camera.read_frame(timeout=0)
                if read.status in ('error', 'closed', 'stale'):
                    raise RuntimeError(read.message or read.status)
                if read.status == 'end':
                    ended = True
                if read.status == 'frame':
                    captured += 1
                    skipped += read.packet.skipped_frames
                    worker.submit(read.packet)
            result = worker.read_result()
            if result.status in ('error', 'closed'):
                raise RuntimeError(result.message or result.status)
            if result.status == 'stale':
                expired += 1
            if result.status == 'result':
                latest_receipt_ns = result.snapshot.packet.received_ns
                ages.append(result.snapshot.packet.age_seconds() * 1000)
            delivery = bridge.poll_action() if mode == 'scheduled' else None
            if delivery is not None:
                if not delivery:
                    raise RuntimeError('Synthetic input workload failed: ' + delivery.error)
                completed += 1
            if result.status == 'result' and not bridge.busy and result.snapshot.packet.sequence >= next_action_sequence:
                sequence = result.snapshot.packet.sequence
                actions = ['Jump', 'Right', 'Right']
                started = time.perf_counter()
                delivery = bridge.send_action(actions) if mode == 'blocking' else bridge.begin_action(actions)
                deliveries.append({'source_sequence': sequence, 'call_ms': round((time.perf_counter() - started) * 1000, 3)})
                if delivery is not None:
                    if not delivery:
                        raise RuntimeError('Synthetic input workload failed: ' + delivery.error)
                    completed += 1
                next_action_sequence += 25
            poll_calls.append((time.perf_counter() - before) * 1000)
            if ended and worker.last_result_sequence >= worker.submitted_sequence and not bridge.busy:
                break
            time.sleep(.005)
        else:
            raise RuntimeError('Recording pipeline exceeded evaluation deadline')
        return {'file': clip['file'], 'source_sha256': clip['sha256'], 'mode': mode,
                'model_initialization_ms': startup_ms, 'captured_frames': captured,
                'skipped_before_captured_frames': skipped, 'fresh_pose_results': len(ages),
                'expired_pose_results': expired, 'receipt_to_pose_read': summary(ages),
                'parent_capture_pose_input_iteration': summary(poll_calls),
                'synthetic_sequences_completed': completed, 'synthetic_dispatches': deliveries}
    finally:
        bridge.close()
        worker.close()
        if camera is not None:
            camera.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--file', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    root = Path(__file__).resolve().parents[1]
    report = {'parent_revision': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ('control/keyboard_bridge.py', 'main.py', 'config/settings.py',
                                            'pose/worker.py', 'camera/camera_stream.py', 'camera/device_capture.py',
                                            'camera/clock.py', 'evaluation/benchmark_input_scheduler.py',
                                            'evaluation/benchmark_pose_worker.py')},
              'model_sha256': hashlib.sha256((root / 'pose_landmarker_lite.task').read_bytes()).hexdigest(),
              'python_version': platform.python_version(), 'windows_version': platform.version(),
              'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'native_input_mocked': True, 'timing_waits_real': True,
              'key_hold_seconds': KEY_HOLD_SECONDS, 'inter_key_gap_seconds': INTER_KEY_GAP_SECONDS,
              'conditions': 'No concurrent regression work; real PyDirectInput event construction with SendInput mocked; '
                            'nominal 5ms polling. Recording runs initialize model before playback, use original checksum-verified '
                            'media at its FPS through production capture/pose workers, and insert a synthetic triple-key workload '
                            'at the first available result at or after each 25-source-frame boundary. '
                            'Pose receipt age guards delivery. No preview, physical camera, gesture classification or live game.',
              'recordings': []}
    with patch('pydirectinput.SendInput', return_value=1), \
         patch('pydirectinput.ctypes.windll.user32.GetKeyState', return_value=0):
        report['dispatch'] = dispatch_measurements(args.samples)
        print(json.dumps({'dispatch': report['dispatch']}), flush=True)
        clips = {clip['file']: clip for clip in json.loads(args.manifest.read_text(encoding='utf-8-sig'))['clips']}
        for name in args.file:
            clip = clips[name]
            path = download_clip(clip, args.data)
            for mode in ('blocking', 'scheduled'):
                result = recording_measurement(clip, path, mode)
                report['recordings'].append(result)
                print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
