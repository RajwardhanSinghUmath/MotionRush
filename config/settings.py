# config/settings.py

# Key bindings
KEY_UP = 'up'
KEY_DOWN = 'down'
KEY_LEFT = 'left'
KEY_RIGHT = 'right'

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS_TARGET = 30

# Gesture thresholds (Will need tuning)
# Displacement thresholds (relative to normalized shoulder width / hip height)
JUMP_THRESHOLD = 0.08  # Decrease in Y (moving up)
SQUAT_THRESHOLD = 0.05 # Increase in Y (moving down)
LEFT_THRESHOLD = 0.1   # Decrease in X
RIGHT_THRESHOLD = 0.1  # Increase in X

# Cooldown timers (in seconds)
ACTION_COOLDOWN = 0.5
