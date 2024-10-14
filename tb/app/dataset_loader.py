from sklearn.model_selection import train_test_split
import os
import shutil
import cv2

# for progress bars (optional, but useful for large datasets)
from tqdm import tqdm


class DataLoader:
    def __init__(
        self,
        image_dir,
        label_dir,
        output_dir,
        augmentations=None,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.test_size = test_size
        self.val_size = val_size
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

    def pad_to_square(self, image, bboxes, orig_width, orig_height, max_dim):
        """Pads the image to square and adjusts bounding boxes."""
        # Calculate padding for width and height
        pad_top = (max_dim - orig_height) // 2
        pad_bottom = max_dim - orig_height - pad_top
        pad_left = (max_dim - orig_width) // 2
        pad_right = max_dim - orig_width - pad_left

        # Apply padding to image
        padded_image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

        # Adjust bounding boxes
        for bbox in bboxes:
            bbox[0] = (bbox[0] * orig_width + pad_left) / max_dim  # x_center
            bbox[1] = (bbox[1] * orig_height + pad_top) / max_dim  # y_center
            bbox[2] = (bbox[2] * orig_width) / max_dim  # width
            bbox[3] = (bbox[3] * orig_height) / max_dim  # height

        return padded_image, bboxes

    def pad_and_save_image_label(
        self, image_path, label_path, output_image_path, output_label_path
    ):
        """Pads the image to a square shape, adjusts the label, and saves both."""
        # Load image and label data
        image = cv2.imread(image_path)
        orig_height, orig_width = image.shape[:2]
        bboxes, class_labels = self.read_yolo_label(label_path)

        # Calculate the max dimension (for square padding)
        max_dim = max(orig_height, orig_width)

        # Pad the image and adjust bounding boxes
        padded_image, bboxes = self.pad_to_square(
            image, bboxes, orig_width, orig_height, max_dim
        )

        # Save padded image
        cv2.imwrite(output_image_path, padded_image)

        # Save adjusted labels
        self.save_yolo_labels(output_label_path, bboxes, class_labels)

    def pad_dataset(self):
        """Pads the dataset images and saves them without redundancy."""
        padded_image_dir = os.path.join(self.output_dir, "padded_images/")
        padded_label_dir = os.path.join(self.output_dir, "padded_labels/")
        os.makedirs(padded_image_dir, exist_ok=True)
        os.makedirs(padded_label_dir, exist_ok=True)

        for img_path, lbl_path in tqdm(
            self.image_label_generator(self.image_paths, self.label_paths),
            desc="Padding images",
        ):
            output_image_path = os.path.join(
                padded_image_dir, os.path.basename(img_path)
            )
            output_label_path = os.path.join(
                padded_label_dir, os.path.basename(lbl_path)
            )
            self.pad_and_save_image_label(
                img_path, lbl_path, output_image_path, output_label_path
            )

        # Update the image and label paths to the new padded ones
        self.image_paths = [
            os.path.join(padded_image_dir, f)
            for f in os.listdir(padded_image_dir)
            if f.endswith(".png")
        ]
        self.label_paths = [
            os.path.join(padded_label_dir, f.replace(".png", ".txt"))
            for f in os.listdir(padded_image_dir)
            if f.endswith(".png")
        ]

    def split_dataset(self):
        """Splits the dataset into training, validation, and test sets."""
        # First split into train+val and test sets
        train_images, test_images, train_labels, test_labels = train_test_split(
            self.image_paths,
            self.label_paths,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        # Further split the train set into train and validation sets
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images,
            train_labels,
            test_size=self.val_size,
            random_state=self.random_state,
        )
        return (
            train_images,
            val_images,
            test_images,
            train_labels,
            val_labels,
            test_labels,
        )

    def copy_images_and_labels(self, images, labels, split):
        """Copies the images and labels to the corresponding directories."""
        img_dst_dir = os.path.join(self.output_dir, f"images/{split}/")
        lbl_dst_dir = os.path.join(self.output_dir, f"labels/{split}/")
        os.makedirs(img_dst_dir, exist_ok=True)
        os.makedirs(lbl_dst_dir, exist_ok=True)

        for img_path, lbl_path in self.image_label_generator(images, labels):
            shutil.copy2(img_path, img_dst_dir)
            shutil.copy2(lbl_path, lbl_dst_dir)

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
        # pad all images and labels
        self.pad_dataset()

        # split dataset
        train_images, val_images, test_images, train_labels, val_labels, test_labels = (
            self.split_dataset()
        )

        # Create directories for saving results
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(self.output_dir, f"images/{split}"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, f"labels/{split}"), exist_ok=True)

        # Copy training and validation images and labels
        self.copy_images_and_labels(train_images, train_labels, split="train")
        self.copy_images_and_labels(val_images, val_labels, split="val")
        self.copy_images_and_labels(test_images, test_labels, split="test")

        # Augment training data
        if self.augmentor is not None:
            self.augment_dataset(train_images, train_labels)
