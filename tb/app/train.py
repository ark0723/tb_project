import DataLoader
import albumentations as A
from ultralytics import YOLO


def train_yolo(image_dir, label_dir, output_dir, augmentations=None):
    DataLoader(
        image_dir=image_dir,
        label_dir=label_dir,
        output_dir=output_dir,
        augmentations=augmentations,
    )()

    # Load the pre-trained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Train the model using a resized image size (640x640)
    model.train(
        data="/dataset/augmented_data.yaml",
        epochs=50,  # Number of epochs
        imgsz=640,  # Image size to resize to (set to 640x640)
        batch=16,  # Batch size
        workers=4,  # Number of workers for data loading
    )


image_dir = "dataset/images/"
label_dir = "dataset/labels/"
output_dir = "dataset/"


augmentation = A.Compose(
    [
        A.RandomCrop(width=2300, height=3300),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(p=0.3),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)
