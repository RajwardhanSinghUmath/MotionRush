"""Collect a continuous, stable, upright reference pose before control starts."""

from collections import deque
from dataclasses import dataclass
import math
from statistics import median

from config.settings import (
    CALIBRATION_SECONDS, CALIBRATION_MIN_SAMPLES, CALIBRATION_MAX_SAMPLE_GAP,
    CALIBRATION_POSITION_RANGE, CALIBRATION_SCALE_RANGE, CALIBRATION_MIN_TORSO_HEIGHT,
    CALIBRATION_MIN_KNEE_ANGLE, MIN_LANDMARK_VISIBILITY, MIN_LANDMARK_PRESENCE,
    CALIBRATION_MAX_LEAN_RATIO,
    JUMP_TORSO_RATIO, SQUAT_TORSO_RATIO, LANE_TORSO_RATIO, LANE_SHOULDER_RATIO,
)


@dataclass(frozen=True)
class CalibrationProfile:
    x_center: float
    y_center: float
    torso_height: float
    shoulder_width: float
    jump_gap: float
    squat_gap: float
    lane_half_width: float

    def fits_frame(self):
        return (0.05 <= self.x_center - self.lane_half_width < self.x_center
                < self.x_center + self.lane_half_width <= 0.95
                and 0.05 <= self.y_center - self.jump_gap < self.y_center
                < self.y_center + self.squat_gap <= 0.95)


class StandingCalibration:
    def __init__(self):
        self.active = False
        self.message = 'Press C to calibrate'
        self.progress = 0.0
        self._samples = deque(maxlen=360)
        self._started_at = None
        self._last_at = None

    def start(self):
        self.active = True
        self._clear_samples()
        self.message = 'Stand upright and still'

    def _clear_samples(self):
        self._samples.clear()
        self._started_at = self._last_at = None
        self.progress = 0.0

    @staticmethod
    def _valid_features(features):
        try:
            x, y = features['hip_center']
            shoulder_x, shoulder_y = features['shoulder_center']
            torso = features['torso_height']
            width = features['shoulder_width']
            confidence = features['confidence']
            aspect = features.get('image_aspect_ratio', 4 / 3)
            angles = [angle for angle in features['knee_angles'] if angle is not None]
            values = (x, y, shoulder_x, shoulder_y, torso, width, confidence, aspect, *angles)
            return (all(math.isfinite(value) for value in values)
                    and 0 <= x <= 1 and 0 <= y <= 1 and 0 <= width <= 1
                    and 0 <= shoulder_x <= 1 and 0 <= shoulder_y <= 1 and aspect > 0
                    and y - shoulder_y >= CALIBRATION_MIN_TORSO_HEIGHT
                    and abs(shoulder_x - x) * aspect <= CALIBRATION_MAX_LEAN_RATIO * torso
                    and CALIBRATION_MIN_TORSO_HEIGHT <= torso <= 0.6
                    and max(MIN_LANDMARK_VISIBILITY, MIN_LANDMARK_PRESENCE) <= confidence <= 1
                    and all(0 <= angle <= 180 for angle in angles)
                    and bool(angles) and min(angles) >= CALIBRATION_MIN_KNEE_ANGLE)
        except (KeyError, TypeError, ValueError):
            return False

    def update(self, features, now):
        if not self.active:
            return None
        if not self._valid_features(features):
            self._clear_samples()
            self.message = 'Stand upright; keep shoulders, hips and a leg visible'
            return None
        if self._last_at is not None:
            if now <= self._last_at or now - self._last_at > CALIBRATION_MAX_SAMPLE_GAP:
                self._clear_samples()
                self.message = 'Tracking interrupted; hold still'
        if self._started_at is None:
            self._started_at = now
        self._last_at = now
        x, y = features['hip_center']
        sample = (x, y, features['torso_height'], features['shoulder_width'])
        self._samples.append(sample)
        scale = median(item[2] for item in self._samples)
        position_limit = CALIBRATION_POSITION_RANGE * scale
        ranges = [max(item[i] for item in self._samples) - min(item[i] for item in self._samples)
                  for i in (0, 1, 2)]
        if ranges[0] > position_limit or ranges[1] > position_limit or ranges[2] > CALIBRATION_SCALE_RANGE * scale:
            self._clear_samples()
            self._samples.append(sample)
            self._started_at = self._last_at = now
            self.message = 'Movement detected; hold still'
            return None
        profile = CalibrationProfile(
            x_center=median(item[0] for item in self._samples),
            y_center=median(item[1] for item in self._samples),
            torso_height=scale,
            shoulder_width=median(item[3] for item in self._samples),
            jump_gap=JUMP_TORSO_RATIO * scale,
            squat_gap=SQUAT_TORSO_RATIO * scale,
            lane_half_width=max(LANE_TORSO_RATIO * scale,
                                LANE_SHOULDER_RATIO * median(item[3] for item in self._samples)),
        )
        if not profile.fits_frame():
            self._clear_samples()
            self.message = 'Move toward the middle of the camera frame'
            return None
        elapsed = now - self._started_at
        self.progress = min(1.0, elapsed / CALIBRATION_SECONDS,
                            len(self._samples) / CALIBRATION_MIN_SAMPLES)
        self.message = f'Hold still: {self.progress:.0%}'
        if elapsed >= CALIBRATION_SECONDS and len(self._samples) >= CALIBRATION_MIN_SAMPLES:
            self.active = False
            self.progress = 1.0
            self.message = 'Calibrated'
            return profile
        return None
