import time
from config.settings import (
    DEFAULT_Y_CENTER,
    VERTICAL_GAP,
    DEFAULT_X_CENTER,
    LANE_HALF_WIDTH,
    ACTION_COOLDOWN,
    OPPOSITE_VERTICAL_COOLDOWN
)

class GestureRecognizer:
    def __init__(self):
        # Ready right away with default center boundaries
        self.is_calibrated = True
        
        # Center positions (0.0 to 1.0)
        self.y_center = DEFAULT_Y_CENTER
        self.vertical_gap = VERTICAL_GAP
        
        self.x_center = DEFAULT_X_CENTER
        self.lane_half_width = LANE_HALF_WIDTH
        
        # Calculated boundary lines
        self.y_upper = self.y_center - self.vertical_gap
        self.y_lower = self.y_center + self.vertical_gap
        self.x_left = self.x_center - self.lane_half_width
        self.x_right = self.x_center + self.lane_half_width
        
        # States
        self.vertical_state = "NEUTRAL"   # "UP", "NEUTRAL", "DOWN"
        self.current_lane = "CENTER"      # "LEFT", "CENTER", "RIGHT"
        
        # Timers
        self.last_up_time = 0.0
        self.last_down_time = 0.0
        self.last_horizontal_time = 0.0
        
        # Display feedback
        self.last_action = "None"
        self.waist_position = None
        self.calibration_frames = []

    def _recalc_lines(self):
        self.y_upper = max(0.05, self.y_center - self.vertical_gap)
        self.y_lower = min(0.95, self.y_center + self.vertical_gap)
        self.x_left = max(0.05, self.x_center - self.lane_half_width)
        self.x_right = min(0.95, self.x_center + self.lane_half_width)

    def calibrate_on_waist(self, features):
        """Sets the center lines directly around player's current standing waist position"""
        if not features:
            return
        hip_x, hip_y = features['hip_center']
        self.y_center = hip_y
        self.x_center = hip_x
        self._recalc_lines()
        self.vertical_state = "NEUTRAL"
        self.current_lane = "CENTER"
        self.last_action = "Centered on Waist"

    def adjust_vertical_gap(self, delta):
        """Increase or decrease distance between the two horizontal lines"""
        self.vertical_gap = max(0.02, min(0.25, self.vertical_gap + delta))
        self._recalc_lines()
        self.last_action = f"Vertical Gap: {self.vertical_gap * 2:.2f}"

    def nudge_vertical(self, delta):
        """Move both horizontal lines up or down"""
        self.y_center = max(0.1, min(0.9, self.y_center + delta))
        self._recalc_lines()

    def nudge_horizontal(self, delta):
        """Move both lane lines left or right"""
        self.x_center = max(0.1, min(0.9, self.x_center + delta))
        self._recalc_lines()

    def reset_lane(self, lane="CENTER"):
        self.current_lane = lane
        self.last_action = f"Lane Reset: {lane}"

    def recognize(self, features):
        """
        UP / DOWN:
          - Waist above upper line -> UP key (Jump)
          - Waist below lower line -> DOWN key (Duck)
          - Returning between lines -> NO KEY (neutral reset)
        
        LEFT / RIGHT:
          - Step past left line -> LEFT key
          - Cross back to center -> RIGHT key
          - Step past right line -> RIGHT key
          - Cross back to center -> LEFT key
        """
        if not features:
            return []

        waist_x, waist_y = features['hip_center']
        self.waist_position = (waist_x, waist_y)
        actions = []
        now = time.time()

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
                    self.last_up_time = now
                    self.last_action = "UP (Above Line)"

        elif waist_y > self.y_lower:
            # Waist went below lower line
            if self.vertical_state != "DOWN" and (now - self.last_down_time) >= ACTION_COOLDOWN:
                # Prevent landing impact after jump from triggering accidental DOWN
                if (now - self.last_up_time) >= OPPOSITE_VERTICAL_COOLDOWN:
                    actions.append("Squat")
                    self.vertical_state = "DOWN"
                    self.last_down_time = now
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
                    self.current_lane = "LEFT"
                    self.last_horizontal_time = now
                    actions.append("Left")
                    self.last_action = "Move Left"
                elif waist_x > self.x_right:
                    self.current_lane = "RIGHT"
                    self.last_horizontal_time = now
                    actions.append("Right")
                    self.last_action = "Move Right"

            elif self.current_lane == "LEFT":
                if waist_x > self.x_right:
                    self.current_lane = "RIGHT"
                    self.last_horizontal_time = now
                    actions.extend(["Right", "Right"])
                    self.last_action = "Left -> Right"
                elif waist_x > self.x_left:
                    # Crossed back to center -> send Right
                    self.current_lane = "CENTER"
                    self.last_horizontal_time = now
                    actions.append("Right")
                    self.last_action = "Return to Center (Right)"

            elif self.current_lane == "RIGHT":
                if waist_x < self.x_left:
                    self.current_lane = "LEFT"
                    self.last_horizontal_time = now
                    actions.extend(["Left", "Left"])
                    self.last_action = "Right -> Left"
                elif waist_x < self.x_right:
                    # Crossed back to center -> send Left
                    self.current_lane = "CENTER"
                    self.last_horizontal_time = now
                    actions.append("Left")
                    self.last_action = "Return to Center (Left)"

        return actions
