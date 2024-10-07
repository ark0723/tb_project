# https://github.com/MinkyungPark/docker_data_science
# https://ebbnflow.tistory.com/340

import os, glob
import numpy as np

# from preprocess import classify_and_preprocess
from preprocessor import ImageProcessor
from boundingbox import BoxDetector, CircleDetector, compare_thresholding
from utils import get_image_list, check_formtype_from_path


if __name__ == "__main__":
    # test1
    # root = os.path.join("/app/image", "original")
    # print(root)
    # scans = os.listdir(root)
    # img_processor = ImageProcessor(root_dir=root)
    # for scan in scans:
    #     img_processor.classify_and_preprocess(file=scan)

    # test2
    root_dir = os.path.join("/app/image")
    min_box_area = 1000
    files = get_image_list(root_dir, pattern="resize*", recursive=True)

    for file in files:
        box_detector = BoxDetector(file, output_dir="/app/image/output")
        circle_detector = CircleDetector(file, output_dir="/app/image/output")
        box_detector.detect(n_box=50, min_box_area=min_box_area)
        positions = circle_detector.detect(
            minDist=20, param1=50, param2=30, minRadius=25, maxRadius=40
        )
        print(positions)
