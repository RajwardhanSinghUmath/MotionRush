"""Measure the actual capture worker on checksum-verified original recordings."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import statistics
import subprocess
import time

from camera.camera_stream import CameraStream
from evaluation.download import download_clip
from pose.estimator import PoseEstimator
from config.settings import CAPTURE_MAX_AGE_SECONDS


class SimulatedBlockedRead:
    def __init__(self, started, clock_start_ns=None):
        self.started = started

    def read_frame(self):
        self.started.value = 1
        time.sleep(60)

    def release(self):
        pass


def run_recording(clip, path, delay_ms):
    estimator = PoseEstimator()
    camera = None
    ages, inference, gaps = [], [], []
    observations = detected = skipped = expired = stale = 0
    previous_ms = None
    started = time.perf_counter()
    try:
        camera = CameraStream(str(path), playback=True)
        deadline = started + 90
        while time.perf_counter() < deadline:
            result = camera.read_frame()
            if result.status in ('error', 'end'):
                if result.status == 'error':
                    raise RuntimeError(result.message)
                break
            if result.status == 'stale':
                stale += 1
                continue
            if result.status != 'frame':
                continue
            packet = result.packet
            ages.append(packet.age_seconds() * 1000)
            if previous_ms is not None:
                gaps.append(packet.timestamp_ms - previous_ms)
            previous_ms = packet.timestamp_ms
            skipped += packet.skipped_frames
            before = time.perf_counter()
            pose = estimator.process_frame(packet.image, packet.timestamp_ms)
            inference.append((time.perf_counter() - before) * 1000)
            detected += bool(pose.pose_landmarks)
            expired += packet.age_seconds() > CAPTURE_MAX_AGE_SECONDS
            observations += 1
            if delay_ms:
                time.sleep(delay_ms / 1000)
        else:
            raise RuntimeError('Recording capture did not end within the evaluation deadline')
        stop_started = time.perf_counter()
        camera.release()
        if not ages:
            raise RuntimeError('No fresh observations available; cannot report capture timing')
        return {'file': clip['file'], 'source_sha256': clip['sha256'],
                'consumer_delay_ms': delay_ms, 'observed_frames': observations,
                'detected_pose_frames': detected, 'skipped_before_delivered_frames': skipped,
                'stale_polls': stale, 'frames_expired_during_inference': expired,
                'age_at_read_median_ms': round(statistics.median(ages), 3),
                'age_at_read_max_ms': round(max(ages), 3),
                'inference_median_ms': round(statistics.median(inference), 3),
                'capture_interval_min_ms': min(gaps) if gaps else None,
                'capture_interval_max_ms': max(gaps) if gaps else None,
                'shutdown_ms': round((time.perf_counter() - stop_started) * 1000, 3),
                'forced_shutdown': camera.forced_shutdown, 'worker_alive_after_shutdown': camera.worker_alive}
    finally:
        if camera is not None:
            camera.release()
        estimator.close()


def blocked_read_check():
    started = mp.get_context('spawn').RawValue('B', 0)
    camera = CameraStream(started, device_factory=SimulatedBlockedRead)
    try:
        until = time.perf_counter() + 10
        while not started.value and time.perf_counter() < until:
            camera.read_frame()
        if not started.value:
            raise RuntimeError('Controlled blocked-read worker did not start')
        before = time.perf_counter()
        result = camera.read_frame(timeout=.05)
        poll_ms = (time.perf_counter() - before) * 1000
        before = time.perf_counter()
        camera.release()
        return {'simulated_driver_fault': True, 'poll_status': result.status,
                'poll_ms': round(poll_ms, 3), 'shutdown_ms': round((time.perf_counter() - before) * 1000, 3),
                'forced_shutdown': camera.forced_shutdown, 'worker_alive_after_shutdown': camera.worker_alive}
    finally:
        camera.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--file', action='append', required=True)
    parser.add_argument('--consumer-delay-ms', type=float, action='append')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fault-check', action='store_true')
    args = parser.parse_args()
    delays = args.consumer_delay_ms or [0, 100]
    if any(not 0 <= delay <= 1000 for delay in delays):
        parser.error('Consumer delay must be between zero and 1000 milliseconds')
    clips = {clip['file']: clip for clip in json.loads(args.manifest.read_text(encoding='utf-8-sig'))['clips']}
    root = Path(__file__).resolve().parents[1]
    report = {'revision': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ('camera/camera_stream.py', 'camera/device_capture.py', 'camera/clock.py', 'config/settings.py', 'evaluation/benchmark_capture.py')},
              'receipt_clock': {'implementation': time.get_clock_info('perf_counter').implementation,
                                'resolution_seconds': time.get_clock_info('perf_counter').resolution},
              'model_sha256': hashlib.sha256((root / 'pose_landmarker_lite.task').read_bytes()).hexdigest(),
              'python_version': platform.python_version(), 'windows_version': platform.version(),
              'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'conditions': 'Original recording paced at media FPS through production capture worker; actual pose inference; controlled consumer delay; no webcam or native keyboard input; receipt age is not exposure age',
              'runs': []}
    for name in args.file:
        clip = clips[name]
        path = download_clip(clip, args.data)
        for delay in delays:
            result = run_recording(clip, path, delay)
            report['runs'].append(result)
            print(json.dumps(result), flush=True)
    if args.fault_check:
        report['controlled_blocked_read'] = blocked_read_check()
        print(json.dumps(report['controlled_blocked_read']), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    mp.freeze_support()
    main()
