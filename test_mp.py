import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from pathlib import Path

def main():
    model_path = Path(__file__).resolve().with_name('pose_landmarker_lite.task')
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    with vision.PoseLandmarker.create_from_options(options):
        print("Detector created successfully!")


if __name__ == '__main__':
    main()
