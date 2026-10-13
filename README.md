# MotionRush

MotionRush uses a webcam and the bundled MediaPipe pose model to translate body movements into arrow-key input for an external runner game. It includes a mirrored preview, threshold guides, a skeleton overlay, and manual recentering. Gesture recognition currently uses position rules; a trained temporal gesture classifier is not yet included.

## Install and run

The controller uses Windows input APIs. Python 3.10 is tested locally; CI checks Windows with Python 3.10 and 3.12. Start in the repository directory, create a virtual environment with `python -m venv .venv`, activate it with `.venv\Scripts\Activate.ps1`, and install with `python -m pip install --require-hashes -r requirements.txt`. Run `python main.py`. The bundled `pose_landmarker_lite.task` model must be present; its path is resolved relative to the application, so launching from another directory also works.

Only `opencv-contrib-python` supplies `cv2` in the locked environment. Avoid installing `opencv-python` or headless variants alongside it. `requirements.in` lists the direct dependencies; regenerate the complete hash-locked dependency set with `uv pip compile --python-version 3.10 --generate-hashes requirements.in -o requirements.txt` after testing dependency changes.

## Using the controller

Use a well-lit area where the camera can see your shoulders and hips. The preview is mirrored. Face the camera with shoulders, hips and at least one leg visible. Focus the preview, press C, and stand upright and still for at least three seconds. Commands are disabled until calibration succeeds; the guides then scale to your standing body dimensions. Move above the upper horizontal line to jump, below the lower line to squat, and across the vertical guides to change lanes. Returning from an outer lane to center sends the opposite lane key.

Input starts disarmed. After calibration, stand upright in the physical lane matching the controller's expected game lane, click the intended game and press F8 to arm that foreground window. F9 pauses, F7 starts calibration, and F10 requests exit; these global hotkeys work while the game has focus. The game must accept arrow keys; configuring a mobile emulator or a game-specific input mapping is separate from running MotionRush. The always-on-top preview does not establish game focus.

Moving focus away from the armed window, changing its title, losing reliable pose tracking or encountering an input failure disables further commands. Returning focus does not resume automatically. Match the expected lane and press F8 again. Arming requires a fresh upright pose and calibration; it cannot clear an input failure. Global control registration must succeed at startup, so a conflicting reserved F7-F10 key stops startup with an actionable error instead of allowing unprotected input. [Control details and validation](docs/armed-input.md) describe the scope and remaining limitations.

While the preview has focus, C starts a new standing calibration and disables input, [ and ] change the vertical gap, W/S move horizontal guides, R resets the controller's assumed lane to center, and Q exits. Calibrate only while standing neutrally and synchronize lane reset with the actual game. Calibration preserves the controller's known lane rather than claiming the game has reset. Threshold distances and controls live in `config/settings.py`.

If native input delivery fails, the controller pauses further commands and displays recovery instructions. A two-stroke lane change updates the expected lane only for completed strokes; a failed stroke can still have affected the game, so this prediction must be checked manually. Center the actual game, focus the preview and press R to synchronize the expected lane. Reset retries any possibly held key release and refuses to resume if release still fails. Return to physical neutral, focus the game and press F8 to arm again. Recalibration does not clear an input failure. Native Windows insertion success does not prove that the game accepted a command.

## Verification

Run `python -B -m unittest discover -s tests -v` for hardware-independent regression tests. Native keyboard injection and the webcam are mocked. Run `python -B test_mp.py` to load the actual model without opening the camera. Run `python -m pip check` to check installed dependencies.

GitHub Actions runs these checks on Windows. A successful check does not establish physical gesture accuracy or game acceptance. The native-input-mocked timing benchmark under `docs/benchmarks` records actual scheduling delays but does not measure physical-movement-to-game latency.

Capture and pose inference run in separate processes, each consuming only the latest available observation. The preview continues polling controls while either worker waits. Returned poses retain their source image, timestamp and receipt time; displaying a cached pose does not repeat recognition or calibration. Frames older than 250 ms since backend receipt cannot authorize input. Receipt age is checked again after inference and before each keystroke. Stalled capture or expired processing disarms input; a fresh frame alone does not resume it. [Capture details](docs/capture-worker.md) and [pose-worker details and original-video measurements](docs/pose-worker.md) explain the limits and shutdown policy.

## Current limitations and troubleshooting

Calibration requires a continuous stable standing interval and validates torso/leg posture. Essential shoulder/hip confidence and framing are checked before issuing gesture features. Oblique camera views or unreliable leg estimates can prevent calibration; use a frontal, well-lit view. Tracking loss now disarms input; robust identity matching and recovery remain under development. Foreground checks operate at window level and cannot distinguish two browser tabs with identical titles or an address field within the same window. Scores, game-over state, and run summaries require a separate game integration.

If the model cannot be found, check the bundled asset at the path reported in the error. If camera capture fails, close other camera applications, check Windows camera access, and confirm the camera index in `CameraStream`. If the game does not respond, confirm keyboard focus and test its arrow-key bindings manually. Camera/model startup and active inference have timeouts; shutdown stops stuck workers after bounded waits. Normal device/model cleanup happens in their owning processes. Forced termination can bypass Python/native cleanup; local recording runs required it when native model close exceeded the 250 ms grace period. A worker error pauses input and closes the app; there is no automatic restart. A small preview may require resizing for readable labels. Startup failures clean up resources already acquired.

Model provenance and project licensing still need to be documented before a release. The dependency lock does not establish redistribution rights for the bundled model or external evaluation videos.
