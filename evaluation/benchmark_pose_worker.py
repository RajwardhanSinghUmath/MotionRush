"""Measure production capture-to-pose transport on verified original videos."""

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
from types import SimpleNamespace

import numpy as np

from camera.camera_stream import CameraStream, CapturedFrame
from camera.clock import now_ns
from evaluation.download import download_clip
from pose.worker import PoseWorker


def summary(values):
    return {'median_ms': round(statistics.median(values), 3), 'max_ms': round(max(values), 3)} if values else None


def wait_ready(worker):
    before = time.perf_counter()
    while time.perf_counter() - before < 12:
        result = worker.read_result()
        if result.status == 'error':
            raise RuntimeError(result.message)
        if worker.ready:
            return round((time.perf_counter() - before) * 1000, 3)
        time.sleep(.005)
    raise RuntimeError('Pose worker did not become ready')


def run_recording(clip, path, delay_ms):
    worker = PoseWorker()
    camera = None
    ages, inference, polls, intervals = [], [], [], []
    captured = submitted = rejected = detected = expired = skipped_capture = 0
    previous_ms = None
    sequences = []
    try:
        warmup_ms = wait_ready(worker)
        camera = CameraStream(str(path), playback=True)
        ended = False
        deadline = time.perf_counter() + 90
        while time.perf_counter() < deadline:
            before = time.perf_counter()
            if not ended:
                read = camera.read_frame(timeout=0)
                if read.status in ('error', 'closed'):
                    raise RuntimeError(read.message or read.status)
                if read.status == 'end':
                    ended = True
                if read.status == 'frame':
                    captured += 1
                    skipped_capture += read.packet.skipped_frames
                    if worker.submit(read.packet):
                        submitted += 1
                    else:
                        rejected += 1
            result = worker.read_result()
            polls.append((time.perf_counter() - before) * 1000)
            if result.status in ('error', 'closed'):
                raise RuntimeError(result.message or result.status)
            if result.status == 'stale':
                expired += 1
            if result.status == 'result':
                snapshot = result.snapshot
                sequences.append(snapshot.packet.sequence)
                ages.append(snapshot.packet.age_seconds() * 1000)
                inference.append(snapshot.inference_ms)
                detected += snapshot.features is not None
                if previous_ms is not None:
                    intervals.append(snapshot.packet.timestamp_ms - previous_ms)
                previous_ms = snapshot.packet.timestamp_ms
            if ended and worker.last_result_sequence >= worker.submitted_sequence:
                break
            time.sleep(delay_ms / 1000 if delay_ms else .005)
        else:
            raise RuntimeError('Dual-worker recording exceeded evaluation deadline')
        if not sequences:
            raise RuntimeError('No fresh poses returned; cannot report successful transport')
        before = time.perf_counter()
        worker.close()
        pose_shutdown_ms = round((time.perf_counter() - before) * 1000, 3)
        before = time.perf_counter()
        camera.release()
        return {'file': clip['file'], 'source_sha256': clip['sha256'], 'consumer_delay_ms': delay_ms,
                'initialization_ms': warmup_ms, 'captured_frames': captured,
                'submitted_frames': submitted, 'rejected_submissions': rejected,
                'skipped_before_captured_frames': skipped_capture,
                'fresh_pose_results': len(sequences), 'reliable_pose_results': detected,
                'expired_pose_results': expired, 'first_result_sequence': sequences[0],
                'last_result_sequence': sequences[-1],
                'missing_results_through_last_submitted_sequence': worker.submitted_sequence - len(sequences),
                'receipt_to_pose_read': summary(ages), 'native_inference': summary(inference),
                'parent_capture_submit_result_poll': summary(polls),
                'pose_source_interval_min_ms': min(intervals) if intervals else None,
                'pose_source_interval_max_ms': max(intervals) if intervals else None,
                'pose_shutdown_ms': pose_shutdown_ms,
                'capture_shutdown_ms': round((time.perf_counter() - before) * 1000, 3),
                'pose_forced_shutdown': worker.forced_shutdown, 'capture_forced_shutdown': camera.forced_shutdown,
                'native_model_close_completed_before_termination': worker.native_close_completed,
                'pose_alive_after_shutdown': worker.worker_alive, 'capture_alive_after_shutdown': camera.worker_alive}
    finally:
        worker.close()
        if camera is not None:
            camera.release()


class ControlledBlockedModel:
    def __init__(self, configuration):
        self.mode, self.active = configuration
        if self.mode == 'initialize':
            self.active.value = 1
            time.sleep(60)

    def process_frame(self, image, timestamp_ms):
        self.active.value = 1
        if self.mode == 'infer':
            time.sleep(60)
        return SimpleNamespace(pose_landmarks=[])

    def extract_landmarks(self, result):
        return None

    def close(self):
        self.active.value = 2
        if self.mode == 'close':
            time.sleep(60)


def fault_check(mode):
    active = mp.get_context('spawn').RawValue('B', 0)
    worker = PoseWorker((mode, active), estimator_factory=ControlledBlockedModel,
                        startup_timeout=3, inference_timeout=.2)
    polls = []
    try:
        if mode != 'initialize':
            wait_ready(worker)
        if mode == 'infer':
            worker.submit(CapturedFrame(np.zeros((24, 32, 3), np.uint8), 1, now_ns(), 1))
        result = None
        before_fault = time.perf_counter()
        if mode != 'close':
            while time.perf_counter() - before_fault < 10:
                before = time.perf_counter()
                result = worker.read_result()
                polls.append((time.perf_counter() - before) * 1000)
                if result.status == 'error':
                    break
                time.sleep(.005)
            if result.status != 'error' or not active.value:
                raise RuntimeError('Fault did not occur inside the controlled model')
        before = time.perf_counter()
        worker.close()
        return {'simulated_model_fault': mode, 'active_fault_confirmed': bool(active.value),
                'watchdog_error': result.message if result else None,
                'parent_poll': summary(polls), 'shutdown_ms': round((time.perf_counter() - before) * 1000, 3),
                'native_model_close_completed_before_termination': worker.native_close_completed,
                'forced_shutdown': worker.forced_shutdown, 'worker_alive_after_shutdown': worker.worker_alive}
    finally:
        worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--file', action='append', required=True)
    parser.add_argument('--consumer-delay-ms', type=float, action='append')
    parser.add_argument('--fault-check', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    delays = args.consumer_delay_ms or [0, 100]
    if any(not 0 <= delay <= 1000 for delay in delays):
        parser.error('Consumer delay must be between zero and 1000 milliseconds')
    clips = {clip['file']: clip for clip in json.loads(args.manifest.read_text(encoding='utf-8-sig'))['clips']}
    root = Path(__file__).resolve().parents[1]
    report = {'parent_revision': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ('pose/worker.py', 'camera/camera_stream.py', 'camera/device_capture.py',
                                            'camera/clock.py', 'config/settings.py', 'evaluation/benchmark_pose_worker.py')},
              'model_sha256': hashlib.sha256((root / 'pose_landmarker_lite.task').read_bytes()).hexdigest(),
              'python_version': platform.python_version(), 'windows_version': platform.version(),
              'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'conditions': 'Model initialization completed before media-FPS-paced capture; first inference included; '
                            'production capture and pose processes; 5ms polling or declared consumer delay; '
                            'original source-image correlation; no preview, gestures, keyboard or webcam; receipt age is not exposure age',
              'runs': []}
    for name in args.file:
        clip = clips[name]
        path = download_clip(clip, args.data)
        for delay in delays:
            result = run_recording(clip, path, delay)
            report['runs'].append(result)
            print(json.dumps(result), flush=True)
    if args.fault_check:
        report['controlled_faults'] = [fault_check(mode) for mode in ('initialize', 'infer', 'close')]
        for result in report['controlled_faults']:
            print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    mp.freeze_support()
    main()
