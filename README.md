# MotionRush

MotionRush uses a webcam and the bundled MediaPipe pose model to translate body movements into arrow-key input for an external runner game. It includes a mirrored preview, threshold guides, a skeleton overlay, and manual recentering. Gesture recognition currently uses position rules; a trained temporal gesture classifier is not yet included.

## Install and run

The controller uses Windows input APIs. Python 3.10 is tested locally; CI checks Windows with Python 3.10 and 3.12. Start in the repository directory, create a virtual environment with `python -m venv .venv`, activate it with `.venv\Scripts\Activate.ps1`, and install with `python -m pip install --require-hashes -r requirements.txt`. Run `python main.py`. The bundled `pose_landmarker_lite.task` model must be present. The current launcher expects the repository as the working directory.

Only `opencv-contrib-python` supplies `cv2` in the locked environment. Avoid installing `opencv-python` or headless variants alongside it. `requirements.in` lists the direct dependencies; regenerate the complete hash-locked dependency set with `uv pip compile --python-version 3.10 --generate-hashes requirements.in -o requirements.txt` after testing dependency changes.

## Using the controller

Use a well-lit area where the camera can see your shoulders and hips. The preview is mirrored. Stand neutrally, focus the preview, and press C to recenter the movement guides. Move above the upper horizontal line to jump, below the lower line to squat, and across the vertical guides to change lanes. Returning from an outer lane to center sends the opposite lane key.

Click the intended game to give it keyboard focus. The current bridge sends commands to the focused application. The game must accept arrow keys; configuring a mobile emulator or a game-specific input mapping is separate from running MotionRush. The always-on-top preview does not establish game focus.

While the preview has focus, C recenters, [ and ] change the vertical gap, W/S move horizontal guides, R resets the controller's assumed lane to center, and Q exits. Recenter only while standing neutrally and synchronize lane reset with the actual game. Threshold distances and controls live in `config/settings.py`.

## Verification

Run `python -B -m unittest discover -s tests -v` for hardware-independent regression tests. Native keyboard injection and the webcam are mocked. Run `python -B test_mp.py` to load the actual model without opening the camera. Run `python -m pip check` to check installed dependencies.

GitHub Actions runs these checks on Windows. A successful check does not establish physical gesture accuracy or game acceptance. The native-input-mocked timing benchmark under `docs/benchmarks` records actual scheduling delays but does not measure physical-movement-to-game latency.

## Current limitations and troubleshooting

Calibration currently uses a single frame, and startup enables threshold rules immediately. Essential shoulder/hip confidence and framing are checked before issuing gesture features. Stable calibration, pause/focus handling, and robust tracking recovery are being developed. Scores, game-over state, and run summaries require a separate game integration.

If the model cannot be found, check the launch directory and bundled asset. If camera capture fails, close other camera applications, check Windows camera access, and confirm the camera index in `CameraStream`. If the game does not respond, confirm keyboard focus and test its arrow-key bindings manually. A small preview may require resizing for readable labels.

Model provenance and project licensing still need to be documented before a release. The dependency lock does not establish redistribution rights for the bundled model or external evaluation videos.
