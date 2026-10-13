# Extended real-motion data

The Weizmann manifest contains original one-hand-waving, jumping-in-place, and sideways-motion recordings for nine subjects. The source archives and each extracted video have verified byte sizes and SHA-256 hashes. The downloader caches verified archives and reads only the selected member as data; archive paths are never extracted to disk. Raw video stays in ignored/local data directories. The KTH manifest identifies two original person15 standing-with-arm-motion recordings. Source pages and publisher use conditions are included.

Download with `python -m evaluation.download --manifest evaluation/manifests/weizmann.json`; replay with `python -m evaluation.replay --manifest evaluation/manifests/weizmann.json`. For KTH, substitute `evaluation/manifests/kth-standing.json`. TLS verification uses the declared certifi CA bundle; certificate checking is never disabled.

Subject partitions are declared before gesture tuning: development daria/denis/eli, validation ido/ira/lena, and held-out lyova/moshe/shahar. Frames from one subject must not cross these partitions. Person15 from KTH is a separate calibration stress case, not an independent multi-subject benchmark.

The current calibration protocol completed on three of nine one-hand-wave recordings (daria, eli, ira). The other six recordings contain only 53-61 frames at their actual 25 FPS, shorter than the required three-second standing interval. They therefore cannot validate completion of that protocol. Both longer KTH recordings produced reliable shoulder/hip features but remained uncalibrated: waving/clapping torso-position excursions exceeded the current stability limit. Zero output from those disabled controllers is not an accuracy success.

The Weizmann webpage describes a deinterlaced frame rate, but the actual downloaded AVI files used here report 25 FPS. Replay uses the actual recording metadata, and derived event annotations must use that same timeline. Jump and lane-event scoring requires inspected event windows and a documented reference pose. A separately supplied offline reference estimate can study recognition, but must never be reported as successful production calibration.

Original sources: https://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html and https://www.csc.kth.se/cvap/actions/. Cite the dataset authors when publishing results, and follow their use conditions. The raw media are not redistributed in this repository.
