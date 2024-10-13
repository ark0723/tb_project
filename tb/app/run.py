# https://github.com/MinkyungPark/docker_data_science
# https://ebbnflow.tistory.com/340

from dataset_loader import DataLoader
import albumentations as A
from train import YoloModel
import os

if __name__ == "__main__":
    image_dir = "/dataset/images/"
    label_dir = "/dataset/labels/"
    output_dir = "/dataset/"

    augmentations = A.Compose(
        [
            A.RandomCrop(width=2350, height=3300),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(p=0.3),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )

    # prepare augmented dataset for YOLO11n
    DataLoader(
        image_dir=image_dir,
        label_dir=label_dir,
        output_dir=output_dir,
        augmentations=augmentations,
    )()

    yolo11 = YoloModel()
    yolo11.update()
    yolo11.predict()
