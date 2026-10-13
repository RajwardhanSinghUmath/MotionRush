# Essential-landmark quality filter

Control features require both shoulders and both hips to have finite, in-frame coordinates and visibility/presence at least 0.5. Missing or malformed confidence is rejected. Unused joints such as wrists do not disable the existing waist-based controller. The minimum accepted confidence is included in the feature dictionary for diagnostics.

On cached actual-model outputs from 15 original UCF101 recordings, the filter retained all 2,717 detected poses across 2,956 frames. This measures preservation on the tested footage, not broad gesture accuracy. Tracking failures that produce no pose remain tracking failures.

The 75-sample fixture combines original footage with explicitly generated lower-body hiding, upper-body cropping, dimming, and blur. The old parser accepted 24 detected poses; the filter accepts 19. Five rejected blurred poses had essential-landmark confidence around 0.001. Other severely transformed frames generally produced no pose. These are controlled perturbations of real recordings, not 75 independent camera sessions.

Regenerate using `python -m evaluation.pose_quality --data PATH`. Sources, file checksums, model checksum, software versions, transforms, and reference thresholds are included in the fixture. The generator uses the actual bundled model and never opens a webcam or injects input. Floating-point model output can vary slightly across environments; assertions validate acceptance behavior rather than bit-for-bit inference coordinates.

The filter prevents low-quality positions from reaching gesture recognition. Stable calibration, temporal confirmation, player identity/reacquisition, accessible pause controls, and target-game validation remain separate requirements.
