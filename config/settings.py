# config/settings.py

# Key bindings for Subway Surfers
KEY_UP = 'up'
KEY_DOWN = 'down'
KEY_LEFT = 'left'
KEY_RIGHT = 'right'

# Explicit input timing instead of PyDirectInput's nested default pauses.
KEY_HOLD_SECONDS = 0.02
INTER_KEY_GAP_SECONDS = 0.03

# Every essential shoulder/hip landmark must be reliable and in frame.
MIN_LANDMARK_VISIBILITY = 0.5
MIN_LANDMARK_PRESENCE = 0.5

# Stable upright calibration; distances scale with neutral torso height.
CALIBRATION_SECONDS = 3.0
CALIBRATION_MIN_SAMPLES = 30
CALIBRATION_MAX_SAMPLE_GAP = 0.25
CALIBRATION_POSITION_RANGE = 0.10
CALIBRATION_SCALE_RANGE = 0.15
CALIBRATION_MIN_TORSO_HEIGHT = 0.07
CALIBRATION_MIN_KNEE_ANGLE = 155.0
CALIBRATION_MAX_LEAN_RATIO = 0.5
JUMP_TORSO_RATIO = 0.20
SQUAT_TORSO_RATIO = 0.35
LANE_TORSO_RATIO = 0.55
LANE_SHOULDER_RATIO = 0.65

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS_TARGET = 30
CAPTURE_POLL_SECONDS = 0.05
CAPTURE_MAX_AGE_SECONDS = 0.25
CAPTURE_STARTUP_SECONDS = 5.0
CAPTURE_JOIN_SECONDS = 0.25
CAPTURE_FORCE_JOIN_SECONDS = 0.5

# Floating Pop-up Window Settings (Always on Top)
WINDOW_WIDTH = 380          # Compact floating pop-up width
WINDOW_HEIGHT = 285         # Compact floating pop-up height
WINDOW_POS_X = 25           # Top-left corner X
WINDOW_POS_Y = 25           # Top-left corner Y

# Two Horizontal Lines for Up / Down
# Jump line is at ~0.67 (where the previous bottom line was)
# Squat line is at ~0.89 (just above the bottom of the camera frame)
DEFAULT_Y_CENTER = 0.78     # Center between 0.67 and 0.89
VERTICAL_GAP = 0.11         # Distance from center to lines (Upper = 0.67, Lower = 0.89)

# Two Vertical Lines for Left / Right Lanes
DEFAULT_X_CENTER = 0.50     # Center of the camera screen
LANE_HALF_WIDTH = 0.14      # Distance from center to lane line (Left = 0.36, Right = 0.64)

# Timing & cooldowns
ACTION_COOLDOWN = 0.22      # Minimum seconds between repeated triggers
OPPOSITE_VERTICAL_COOLDOWN = 0.35  # Cooldown between UP and DOWN to prevent jump-landing bounce
