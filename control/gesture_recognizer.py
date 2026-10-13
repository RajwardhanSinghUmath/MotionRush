import math
import time
from control.calibration import StandingCalibration
from control.delivery import DeliveryResult
from config.settings import (
    DEFAULT_Y_CENTER, VERTICAL_GAP, DEFAULT_X_CENTER, LANE_HALF_WIDTH,
    ACTION_COOLDOWN, OPPOSITE_VERTICAL_COOLDOWN,
)


class GestureRecognizer:
    def __init__(self):
        self.is_calibrated = False
        self.calibration = StandingCalibration()
        self.calibration_profile = None
        self.status_message = self.calibration.message
        self.y_center = DEFAULT_Y_CENTER
        self.jump_gap = self.squat_gap = VERTICAL_GAP
        self.x_center = DEFAULT_X_CENTER
        self.lane_half_width = LANE_HALF_WIDTH
        self._recalc_lines()
        self.vertical_state = "NEUTRAL"
        self.current_lane = "CENTER"
        self.last_up_time = self.last_down_time = self.last_horizontal_time = float('-inf')
        self.last_action = "None"
        self.waist_position = None
        self.input_fault = False
        self.input_uncertain = False
        self.pending_actions = ()
        self._pending_at = None
        self._needs_neutral = False
        self._skip_next_action = False

    def prepare_resume(self):
        """Resume from an explicitly verified matching pose, without a start pulse."""
        if self.input_fault or self.pending_actions:
            raise ValueError('Resolve input delivery before resuming')
        self.vertical_state = 'NEUTRAL'
        self._needs_neutral = False
        self._skip_next_action = True

    @property
    def vertical_gap(self):
        return (self.jump_gap + self.squat_gap) / 2

    def _recalc_lines(self):
        self.y_upper = self.y_center - self.jump_gap
        self.y_lower = self.y_center + self.squat_gap
        self.x_left = self.x_center - self.lane_half_width
        self.x_right = self.x_center + self.lane_half_width

    def start_calibration(self):
        """Disable actions and collect a fresh continuous standing reference."""
        self.is_calibrated = False
        self.calibration_profile = None
        self.calibration.start()
        self.status_message = self.calibration.message
        self.vertical_state = "NEUTRAL"
        self.last_up_time = self.last_down_time = self.last_horizontal_time = float('-inf')
        self.last_action = "Calibrating"
        if self.pending_actions:
            self.input_fault = self.input_uncertain = True
            self.pending_actions = ()
        # Calibration does not assert that the external game changed lanes.

    def _apply_profile(self, profile):
        self.calibration_profile = profile
        self.x_center, self.y_center = profile.x_center, profile.y_center
        self.jump_gap, self.squat_gap = profile.jump_gap, profile.squat_gap
        self.lane_half_width = profile.lane_half_width
        self._recalc_lines()
        self.vertical_state = "NEUTRAL"
        self.is_calibrated = True
        self.last_action = "Calibrated"
        self.status_message = 'Input paused; center game, press R' if self.input_fault else 'Ready'

    def _place_boundaries(self, x, y, jump, squat, lane):
        values = (x, y, jump, squat, lane)
        if not all(math.isfinite(value) for value in values):
            return False
        if not (0.05 <= x - lane < x < x + lane <= 0.95
                and 0.05 <= y - jump < y < y + squat <= 0.95):
            self.last_action = "Adjustment outside frame"
            return False
        self.x_center, self.y_center = x, y
        self.jump_gap, self.squat_gap, self.lane_half_width = jump, squat, lane
        self._recalc_lines()
        return True

    def adjust_vertical_gap(self, delta):
        if not math.isfinite(delta):
            return False
        return self._place_boundaries(
            self.x_center, self.y_center,
            max(0.02, min(0.25, self.jump_gap + delta)),
            max(0.02, min(0.25, self.squat_gap + delta)), self.lane_half_width,
        )

    def nudge_vertical(self, delta):
        return self._place_boundaries(self.x_center, self.y_center + delta,
                                      self.jump_gap, self.squat_gap, self.lane_half_width)

    def nudge_horizontal(self, delta):
        return self._place_boundaries(self.x_center + delta, self.y_center,
                                      self.jump_gap, self.squat_gap, self.lane_half_width)

    def reset_lane(self, lane="CENTER"):
        if lane not in ("LEFT", "CENTER", "RIGHT"):
            raise ValueError('Unknown lane')
        self._needs_neutral = self.input_fault or bool(self.pending_actions)
        self.current_lane = lane
        self.pending_actions = ()
        self.input_fault = self.input_uncertain = False
        self.vertical_state = 'NEUTRAL'
        self.status_message = 'Ready' if self.is_calibrated else self.calibration.message
        self.last_action = f"Lane Reset: {lane}"

    def acknowledge_delivery(self, result):
        """Commit only completed strokes; native insertion is not game acceptance."""
        if not self.pending_actions or not isinstance(result, DeliveryResult) or result.requested != self.pending_actions:
            self.input_fault = self.input_uncertain = True
            self.status_message = 'Input paused; center game, press R'
            raise ValueError('Delivery acknowledgement does not match pending actions')
        lanes = ('LEFT', 'CENTER', 'RIGHT')
        for action in result.completed:
            if action == 'Jump':
                self.last_up_time = self._pending_at
            elif action == 'Squat':
                self.last_down_time = self._pending_at
            else:
                offset = -1 if action == 'Left' else 1
                index = max(0, min(2, lanes.index(self.current_lane) + offset))
                self.current_lane = lanes[index]
                self.last_horizontal_time = self._pending_at
        self.pending_actions = ()
        self._pending_at = None
        if result:
            self.last_action = 'Sent: ' + ', '.join(result.completed)
        else:
            self.input_fault = True
            self.input_uncertain = result.uncertain
            self.last_action = 'Input failed'
            self.status_message = 'Input paused; center game, press R'

    def recognize(self, features, timestamp_ms=None, actions_enabled=True):
        """Use measured frame time, suppressing actions until calibration completes."""
        now = time.monotonic() if timestamp_ms is None else timestamp_ms / 1000
        if not math.isfinite(now) or now < 0:
            raise ValueError('Frame time must be finite and nonnegative')
        if not features:
            self.waist_position = None
            if self.calibration.active:
                self.calibration.update(None, now)
                self.status_message = self.calibration.message
            else:
                self.status_message = "Tracking lost" if self.is_calibrated else "Press C to calibrate"
            if self.input_fault:
                self.status_message = 'Input paused; center game, press R'
            return []
        waist_x, waist_y = features['hip_center']
        if not (math.isfinite(waist_x) and math.isfinite(waist_y)
                and 0 <= waist_x <= 1 and 0 <= waist_y <= 1):
            return self.recognize(None, timestamp_ms)
        self.waist_position = (waist_x, waist_y)
        if not self.is_calibrated:
            if self.calibration.active:
                profile = self.calibration.update(features, now)
                self.status_message = self.calibration.message
                if profile is not None:
                    self._apply_profile(profile)
            return []
        if self.input_fault:
            self.status_message = 'Input paused; center game, press R'
            return []
        if self.pending_actions:
            self.status_message = 'Waiting for input delivery'
            return []
        if not actions_enabled or self._skip_next_action:
            self._skip_next_action = False
            self.vertical_state = 'NEUTRAL'
            return []
        if self._needs_neutral:
            if not (self.x_left <= waist_x <= self.x_right and self.y_upper <= waist_y <= self.y_lower):
                self.status_message = 'Return to neutral before resuming'
                return []
            self._needs_neutral = False
            self.vertical_state = 'NEUTRAL'
            self.status_message = 'Ready'
            return []
        self.status_message = "Ready"
        actions = []

        # =========================================================
        # 1. VERTICAL TRACKING (TWO HORIZONTAL LINES)
        # =========================================================
        # Upper Line: in screen coordinates, smaller Y is ABOVE
        if waist_y < self.y_upper:
            # Waist went above upper line
            if self.vertical_state != "UP" and (now - self.last_up_time) >= ACTION_COOLDOWN:
                # Prevent bounce-triggering DOWN right after UP
                if (now - self.last_down_time) >= OPPOSITE_VERTICAL_COOLDOWN:
                    actions.append("Jump")
                    self.vertical_state = "UP"
                    self.last_action = "UP (Above Line)"

        elif waist_y > self.y_lower:
            # Waist went below lower line
            if self.vertical_state != "DOWN" and (now - self.last_down_time) >= ACTION_COOLDOWN:
                # Prevent landing impact after jump from triggering accidental DOWN
                if (now - self.last_up_time) >= OPPOSITE_VERTICAL_COOLDOWN:
                    actions.append("Squat")
                    self.vertical_state = "DOWN"
                    self.last_action = "DOWN (Below Line)"

        else:
            # Waist is between upper and lower lines (NEUTRAL)
            # NO KEY ON COMING BACK!
            self.vertical_state = "NEUTRAL"

        # =========================================================
        # 2. HORIZONTAL TRACKING (TWO VERTICAL LINES)
        # =========================================================
        if (now - self.last_horizontal_time) >= ACTION_COOLDOWN:
            if self.current_lane == "CENTER":
                if waist_x < self.x_left:
                    actions.append("Left")
                    self.last_action = "Move Left"
                elif waist_x > self.x_right:
                    actions.append("Right")
                    self.last_action = "Move Right"

            elif self.current_lane == "LEFT":
                if waist_x > self.x_right:
                    actions.extend(["Right", "Right"])
                    self.last_action = "Left -> Right"
                elif waist_x > self.x_left:
                    # Crossed back to center -> send Right
                    actions.append("Right")
                    self.last_action = "Return to Center (Right)"

            elif self.current_lane == "RIGHT":
                if waist_x < self.x_left:
                    actions.extend(["Left", "Left"])
                    self.last_action = "Right -> Left"
                elif waist_x < self.x_right:
                    # Crossed back to center -> send Left
                    actions.append("Left")
                    self.last_action = "Return to Center (Left)"

        if actions:
            self.pending_actions = tuple(actions)
            self._pending_at = now
        return actions
