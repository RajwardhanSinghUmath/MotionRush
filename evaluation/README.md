# Recorded-video evaluation

The manifest identifies 15 real UCF101 clips from the University of Central Florida's original CRCV server, with exact byte sizes and SHA-256 checksums. It covers three source groups each of BodyWeightSquats, JumpingJack, JumpRope, TaiChi, and WalkingWithDog. Videos are downloaded locally and are not committed to this repository. Consult the dataset publisher for its use conditions and original citation: https://www.crcv.ucf.edu/research/data-sets/ucf101/.

Run `python -m evaluation.download` from the repository directory to acquire and verify them. Run `python -m evaluation.replay` to process the videos using the actual bundled pose model and current recognizer. The evaluator opens recorded videos, never a webcam, and never imports or calls the keyboard bridge. Its output consists of per-frame landmarks/features/decisions and a summary under `evaluation/runs/latest`.

Use `--class-name BodyWeightSquats` to select a class, `--data PATH` to reuse verified downloads, and `--output PATH` to retain separate experiment results. Keep revisions and model hashes with every comparison. Playback timestamps use each recording's known FPS; this is appropriate for a recording and differs from live camera receipt timing. Cooldowns run against that playback clock rather than the speed of offline inference.

The initial result in `docs/evaluation/uncalibrated-baseline.json` processed 2,956 frames. It used the original uncalibrated default thresholds, so it is a failure-discovery baseline, not a recommended gameplay configuration. The first two squat recordings emitted Jump without Squat. The third emitted both vertical and lateral actions. All three TaiChi clips emitted Jump. The third walking clip produced poses in only 40 of 240 frames, showing a useful tracking-loss case. These findings warrant calibration and tracking work; they do not by themselves establish event-level precision or recall.

UCF101's class names label whole clips, not individual game commands or neutral calibration intervals. A squat clip can begin crouched, and lateral body shifts during a squat are not necessarily intentional lane changes. Manually inspect footage and annotate event windows before measuring command accuracy. Split tuning and evaluation by source group, not by adjacent frames, and keep these stress tests separate from target-camera/game validation.

The replay CLI accepts `--annotations FILE` with a JSON mapping from video filenames to lists of `{ "action": "Squat", "start_ms": 100, "end_ms": 300 }`. Each annotated window expects one command. Duplicate commands and commands outside the expected windows count as false positives, while unmatched windows count as misses. Annotation files must cover the intended evaluation interval completely; clip-level labels alone must not be converted into guessed event labels.

Raw-video and local-run directories are ignored. Public reports should contain provenance, conditions, and limitations, along with measured results. A successful unit test, zero commands from a disabled recognizer, or good behavior on these clips does not prove a complete controller for real-world gameplay.

Run `python -m evaluation.benchmark_input` to reproduce input-stall measurements. It mocks native SendInput while retaining real key-hold and gap sleeps; it does not inject keys or measure game reaction time.
