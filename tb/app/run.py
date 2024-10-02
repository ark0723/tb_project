# https://github.com/MinkyungPark/docker_data_science
# https://ebbnflow.tistory.com/340

import os, glob

# from preprocess import classify_and_preprocess
from preprocessor import ImageProcessor
from boundingbox import BoxDetector, compare_thresholding
from deskew import deskew_image
import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    # # test1
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

    # test4
    file_dir = os.path.join("/app/image/TS04", "Scan2022-01-10_105200_035")
    file = "resize_Scan2022-01-10_105200_035.png"
    detector = BoxDetector(file_dir, file, output_dir="/app/image/output")
    min_box_area = 1000
    n_box = 15

    box_dict = detector.detect_boxes(
        n_box,
        min_box_area,
        contour=(cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE),
    )
