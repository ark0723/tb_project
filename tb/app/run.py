# https://github.com/MinkyungPark/docker_data_science
# https://ebbnflow.tistory.com/340

import os, glob

# from preprocess import classify_and_preprocess
from preprocessor import ImageProcessor
from boundingbox import BoxDetector, compare_thresholding
from utils import get_image_list, check_formtype_from_path
import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    # test1
    # root = os.path.join("/app/image", "original")
    # print(root)
    # scans = os.listdir(root)

    # img_processor = ImageProcessor(root_dir=root)
    # for scan in scans:
    #     img_processor.classify_and_preprocess(file=scan)

    # # test2
    # root = os.path.join("/app/image")
    # for path, dirs, files in os.walk(root):
    #     for dir in dirs:
    #         if dir.startswith("TS"):

    # # test3
    # file_dir = os.path.join("/app/image/TS04", "Scan2022-01-10_105200_035")
    # file = "resize_Scan2022-01-10_105200_035.png"
    # compare_thresholding(file_dir, file, output_dir="/app/image/output")

    # box_dict = detector.detect_boxes(
    #     n_box,
    #     min_box_area,
    #     contour=(cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE),
    # )

    # test 5: process multiple files in parallel
    # root_dir = os.path.join("/app/image", "original")
    # files = os.listdir(root_dir)
    # processor = ImageProcessor(root_dir=root_dir)
    # processor.process_files_in_parallel(files, max_workers=4)

    # test6
    root_dir = os.path.join("/app/image")
    files = get_image_list(root_dir, pattern="resize*", recursive=True)
    min_box_area = 1000
    for file in files:
        detector = BoxDetector(file, output_dir="/app/image/output")
        detector.detect_boxes(
            n_box=50,
            min_box_area=min_box_area,
        )
