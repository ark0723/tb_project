# https://github.com/MinkyungPark/docker_data_science
# https://ebbnflow.tistory.com/340

import os, glob
import numpy as np

# from preprocess import classify_and_preprocess
# from preprocessor import ImageProcessor
# from boundingbox import BoxDetector, CircleDetector, compare_thresholding
# from utils import get_image_list, check_formtype_from_path, move_files_to_parent
from dataset_loader import DataLoader
import albumentations as A
from tb.app.train import train_yolo

if __name__ == "__main__":
    image_dir = "/dataset/images/"
    label_dir = "/dataset/labels/"
    output_dir = (
        "/dataset/"  # Output directory where the split and augmented data will be saved
    )

    augmentations = A.Compose(
        [
            A.RandomCrop(width=2350, height=3300),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(p=0.3),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )

    dataloader = DataLoader(
        image_dir=image_dir,
        label_dir=label_dir,
        output_dir=output_dir,
        augmentations=augmentations,
    )

    dataloader()
