import os
import shutil
from sklearn.model_selection import train_test_split
import cv2


class DataLoader:
    def __init__(
        self,
        image_dir,
        label_dir,
        output_dir,
        augmentations=None,
        test_size=0.2,
        random_state=42,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state

        self.augmentor = augmentations

        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".png")
        ]
        self.label_paths = [
            os.path.join(label_dir, f.replace(".png", ".txt"))
            for f in os.listdir(image_dir)
            if f.endswith(".png")
        ]

    def image_label_generator(self, images, labels):
        for img, lbl in zip(images, labels):
            yield img, lbl

    def split_dataset(self):
        """Splits the dataset into training and validation sets."""
        train_images, val_images, train_labels, val_labels = train_test_split(
            self.image_paths,
            self.label_paths,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return train_images, val_images, train_labels, val_labels

    def copy_images_and_labels(self, images, labels, split):
        """Copies images and labels to train/val directories."""
        for img, lbl in self.image_label_generator(images, labels):
            shutil.copy2(img, os.path.join(self.output_dir, f"images/{split}/"))
            shutil.copy2(lbl, os.path.join(self.output_dir, f"labels/{split}/"))

    def read_yolo_label(self, label_path):
        """Reads YOLO label and returns bounding boxes and class labels."""
        bboxes = []
        class_labels = []
        with open(label_path, "r") as file:
            for line in file.readlines():
                class_id, x_center, y_center, width, height = map(
                    float, line.strip().split()
                )
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(int(class_id))
        return bboxes, class_labels

    def save_yolo_labels(self, label_path, bboxes, class_labels):
        """Saves the augmented YOLO labels."""
        with open(label_path, "w") as f:
            for bbox, class_id in zip(bboxes, class_labels):
                bbox = [str(x) for x in bbox]
                line = f"{class_id} " + " ".join(bbox) + "\n"
                f.write(line)

    def augment_image_and_label(
        self, image_path, label_path, output_image_path, output_label_path
    ):
        """Augments image and corresponding labels."""
        image = cv2.imread(image_path)
        h, w = image.shape[:2]

        # Read yolo label (bounding boxes in YOLO format)
        bboxes, class_labels = self.read_yolo_label(label_path)

        # perform augmentation
        augmented = self.augmentor(
            image=image, bboxes=bboxes, class_labels=class_labels
        )

        # save augmented image nad labels
        cv2.imwrite(output_image_path, augmented["image"])
        self.save_yolo_labels(
            output_label_path, augmented["bboxes"], augmented["class_labels"]
        )

    def augment_dataset(self, train_images, train_labels):
        """Applies augmentation to the training dataset and saves results."""
        augmented_image_dir = os.path.join(self.output_dir, "images/train_augmented/")
        augmented_label_dir = os.path.join(self.output_dir, "labels/train_augmented/")
        os.makedirs(augmented_image_dir, exist_ok=True)
        os.makedirs(augmented_label_dir, exist_ok=True)

        for img_path, lbl_path in self.image_label_generator(
            train_images, train_labels
        ):
            augmented_img_path = os.path.join(
                augmented_image_dir, os.path.basename(img_path)
            )
            augmented_lbl_path = os.path.join(
                augmented_label_dir, os.path.basename(lbl_path)
            )
            self.augment_image_and_label(
                img_path, lbl_path, augmented_img_path, augmented_lbl_path
            )

    def __call__(self):
        # split dataset
        train_images, val_images, train_labels, val_labels = self.split_dataset()

        # Create directories for saving results
        os.makedirs(os.path.join(self.output_dir, "images/train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images/val"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels/train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels/val"), exist_ok=True)

        # Copy training and validation images and labels
        self.copy_images_and_labels(train_images, train_labels, split="train")
        self.copy_images_and_labels(val_images, val_labels, split="val")

        # Augment training data
        if self.augmentor is not None:
            self.augment_dataset(train_images, train_labels)
