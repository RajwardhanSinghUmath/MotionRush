# config/settings.py

# Key bindings for Subway Surfers
KEY_UP = 'up'
KEY_DOWN = 'down'
KEY_LEFT = 'left'
KEY_RIGHT = 'right'

# Explicit input timing instead of PyDirectInput's nested default pauses.
KEY_HOLD_SECONDS = 0.02
INTER_KEY_GAP_SECONDS = 0.03

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS_TARGET = 30

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
