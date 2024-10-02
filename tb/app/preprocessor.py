import os
import logging
import cv2
import inspect
import numpy as np
from data import positions_map, rules_map
from pupil_apriltags import Detector
from concurrent.futures import ThreadPoolExecutor
from utils import (
    create_folders,
    move_to_folder,
    save_image,
)

# Set up logging
logging.basicConfig(level=logging.INFO)


class ImageProcessor:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.detector = Detector()

    def classify_and_preprocess(self, file):
        """Classify image based on AprilTag and preprocess it."""

        print(f"Processing {file}")

        img_path = os.path.join(self.root_dir, file)
        if not os.path.exists(img_path):
            logging.error(f"File does not exist: {img_path}")
            return

        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img_gray is None:
            logging.error(f"Invalid image {img_path}")
            return

        # AprilTag detection
        result = self.detector.detect(img_gray)

        if result and len(result) > 0:
            tag_id = result[0].tag_id
            new_folder_path, form_type = self.get_form_folder(tag_id)

            if new_folder_path:
                move_to_folder(img_path, new_folder_path, file)
                self.preprocess(file, new_folder_path, img_gray, result)
            else:
                logging.error(f"Tag ID {tag_id} does not map to a known form.")
        else:
            logging.warning(f"No AprilTags detected in {file}")

    def preprocess(self, file, new_folder_path, img_gray, result, save=True):
        """Preprocess the image."""
        # Create a directory for the file's preprocessing results
        name = os.path.splitext(file)[0]
        file_dir = os.path.join(new_folder_path, name)
        os.makedirs(file_dir, exist_ok=True)
        # Create subdirectories for processed images
        create_folders(["rect", "small"], file_dir)

        h, w = img_gray.shape[:2]
        # Map detected tags by their IDs
        detected_tags = {r.tag_id: r for r in result}
        # Mapping of specific tag IDs to their intended positions
        form_type = new_folder_path.split("/")[-1]
        tag_positions = self.get_position_by_form_type(form_type)
        # Reverse the mapping to make it easier to determine the tags' positions
        detected_positions = {
            tag_positions[tag_id]: tag
            for tag_id, tag in detected_tags.items()
            if tag_id in tag_positions
        }

        # Rotation check and transformation
        if self.check_for_rotation(result):
            img_gray = self.rotate(img_gray)

        # Proceed with transformation
        resized_img = self.resize(
            detected_positions, img_gray, h, w, file_dir, file, save
        )
        if resized_img is not None:
            self.extract_bounding_boxes(resized_img, file_dir, form_type, save=save)

    def resize(self, detected_positions, img_gray, h, w, file_dir, file, save=True):
        """Handle the affine or perspective transformation based on detected positions."""
        if len(detected_positions) < 3:
            logging.error(
                f"Not enough tags detected for transformation in {file}. Minimum 3 required, but found {len(detected_positions)}."
            )
            return None  # Exit early if there are not enough tags
        pts1, pts2 = self.get_transformation_points(detected_positions)

        if pts1 is not None and pts2 is not None:
            return self.apply_transform(
                pts1, pts2, img_gray, h, w, file_dir, file, save
            )
        else:
            logging.error(f"Unable to identify appropriate tag combination in {file}")

    def apply_transform(self, pts1, pts2, img_gray, h, w, file_dir, file, save=True):
        """Apply affine or perspective transformation and save the image."""
        if len(pts1) == 4:
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(img_gray, M, (w, h))
        elif len(pts1) == 3:
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(img_gray, M, (w, h))
        else:
            logging.error(f"Invalid number of transformation points: {len(pts1)}")
            raise ValueError("Invalid number of points for transformation")

        if save:
            name = file.split(".")[0]
            f_name = "resize_" + name + ".png"
            cv2.imwrite(os.path.join(file_dir, f_name), dst)
            logging.info(f"Saved transformed image: {f_name} | (h, w): {dst.shape[:2]}")

        return dst

    def extract_bounding_boxes(
        self, img, file_dir, form_type, margin=(0, 0, 0, 0), save=True
    ):
        """Extract and save bounding boxes."""
        rec_dir = os.path.join(file_dir, "rect")
        small_dir = os.path.join(file_dir, "small")

        positions = positions_map[form_type]
        rules = rules_map.get(form_type, [])

        rects = []
        for key, (x, y, w, h) in positions.items():
            for rule in rules:
                width_range, properties = rule[0], rule[1:]
                if width_range[0] <= w <= width_range[1]:
                    conditions_met = all(
                        (
                            condition(x, y, h)
                            if len(inspect.signature(condition).parameters) == 3
                            else (
                                condition(x, y)
                                if len(inspect.signature(condition).parameters) == 2
                                else condition(x)
                            )
                        )
                        for condition in properties[1:]
                    )
                    if conditions_met:
                        rects.append((*properties[0], (x, y, w, h)))
                        if save:
                            save_image(img[y : y + h, x : x + w], rec_dir, f"{key}.jpg")
        rects = sorted(rects, key=lambda kv: kv[0])

        # cut out to small boxes
        small_rects = {}
        num_elements = 0

        # unpack margin
        x_margin, y_margin, w_margin, h_margin = margin
        for rect in rects:
            # rect = (id, 가로칸수, 세로칸수, (x,y,w,h)) or rect = (id, 가로칸수, (x,y,w,h))
            if len(rect) == 4:
                idx, cols, rows, (x, y, w, h) = rect
                for j in range(cols):
                    for k in range(rows):
                        small_w = int(w / cols)
                        small_h = int(h / rows)
                        small_x, small_y = (
                            x + j * small_w + x_margin,
                            y + k * small_h + y_margin,
                        )
                        # Ensure width, height is at least 1 to prevent error
                        small_w = max(small_w - w_margin, 1)
                        small_h = max(small_h - h_margin, 1)
                        if save:
                            if small_h > 0 and small_w > 0:
                                save_image(
                                    img[
                                        small_y : small_y + small_h,
                                        small_x : small_x + small_w,
                                    ],
                                    small_dir,
                                    f"{num_elements}.png",
                                )

                            else:
                                print(
                                    f"Skipping saving for rect {num_elements}: invalid dimensions ({small_w}, {small_h})"
                                )
                        small_rects[num_elements + 1] = (
                            small_x,
                            small_y,
                            small_w,
                            small_h,
                        )
                        num_elements += 1
            elif len(rect) == 3:
                idx, cols, (x, y, w, h) = rect
                small_w = int(w / cols)
                for j in range(cols):
                    small_x = x + j * small_w + x_margin
                    small_h = h - h_margin
                    small_w += w_margin
                    small_y = y + y_margin
                    if save:
                        if small_h > 0 and small_w > 0:
                            save_image(
                                img[
                                    small_y : small_y + small_h,
                                    small_x : small_x + small_w,
                                ],
                                small_dir,
                                f"{num_elements}.png",
                            )
                        else:
                            print(
                                f"Skipping saving for rect {num_elements}: invalid dimensions ({small_w}, {small_h})"
                            )
                    small_rects[num_elements + 1] = (small_x, small_y, small_w, small_h)
                    num_elements += 1
        return small_rects

    # Utility methods for file handling, folder creation, etc.
    def get_form_folder(self, tag_id):
        """Map tag_id to the correct folder and form type."""
        # Dictionary to map tag IDs to folder names
        tag_to_folder = {
            (110, 111, 112, 113): "TS01",
            (210, 211, 212, 213): "TS02A",
            (220, 221, 222, 223): "TS02B",
            (310, 311, 312, 313): "TS03",
            (410, 411, 412, 413): "TS04",
            (510, 511, 512, 513): "TS05A",
            (520, 521, 522, 523): "TS05B",
            (60, 61, 62, 63): "TS06",
        }

        for tag_ids, folder in tag_to_folder.items():
            if tag_id in tag_ids:
                # Returning the corresponding folder and the form type (folder name)
                return os.path.join(os.path.dirname(self.root_dir), folder), folder
        return None, None  # If no match is found

    def check_for_rotation(self, result):
        """Check if the detected tags require image rotation."""
        return len(result) > 1 and (result[0].tag_id > result[1].tag_id)

    def rotate(self, img, angle=180):
        """Rotate the image by a given angle."""
        h, w = img.shape[:2]
        m180 = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        return cv2.warpAffine(img, m180, (w, h))

    def get_transformation_points(self, detected_positions):
        """Determine transformation points based on detected positions."""

        if len(detected_positions) == 4:
            pts1 = np.float32(
                [
                    detected_positions["top_left"].center,
                    detected_positions["top_right"].center,
                    detected_positions["bottom_left"].center,
                    detected_positions["bottom_right"].center,
                ]
            )
            pts2 = np.float32([[120, 160], [2300, 160], [120, 3300], [2300, 3300]])
        elif len(detected_positions) == 3:
            if "top_left" not in detected_positions:

                pts1 = np.float32(
                    [
                        detected_positions["top_right"].center,
                        detected_positions["bottom_left"].center,
                        detected_positions["bottom_right"].center,
                    ]
                )
                pts2 = np.float32([[2300, 160], [120, 3300], [2300, 3300]])

            elif "top_right" not in detected_positions:
                pts1 = np.float32(
                    [
                        detected_positions["top_left"].center,
                        detected_positions["bottom_left"].center,
                        detected_positions["bottom_right"].center,
                    ]
                )
                pts2 = np.float32([[120, 160], [120, 3300], [2300, 3300]])

            elif "bottom_left" not in detected_positions:
                pts1 = np.float32(
                    [
                        detected_positions["top_left"].center,
                        detected_positions["top_right"].center,
                        detected_positions["bottom_right"].center,
                    ]
                )
                pts2 = np.float32([[120, 160], [2300, 160], [2300, 3300]])

            elif (
                "top_left" in detected_positions
                and "top_right" in detected_positions
                and "bottom_left" in detected_positions
            ):
                pts1 = np.float32(
                    [
                        detected_positions["top_left"].center,
                        detected_positions["top_right"].center,
                        detected_positions["bottom_left"].center,
                    ]
                )
                pts2 = np.float32([[120, 160], [2300, 160], [120, 3300]])
        else:
            logging.error("Not enough tags detected for transformation")
            return
        return pts1, pts2

    def get_position_by_form_type(self, form_type):
        """Get the expected position of AprilTags based on form type."""
        form_to_tag = {
            "TS01": (110, 111, 112, 113),
            "TS02A": (210, 211, 212, 213),
            "TS02B": (220, 221, 222, 223),
            "TS03": (310, 311, 312, 313),
            "TS04": (410, 411, 412, 413),
            "TS05A": (510, 511, 512, 513),
            "TS05B": (520, 521, 522, 523),
            "TS06": (610, 611, 612, 613),
        }

        if form_type not in form_to_tag:
            logging.error(f"Invalid form type: {form_type}")
            return {}

        position = {
            form_to_tag[form_type][0]: "top_left",
            form_to_tag[form_type][1]: "top_right",
            form_to_tag[form_type][2]: "bottom_left",
            form_to_tag[form_type][3]: "bottom_right",
        }

        return position

    def process_files_in_parallel(self, files, max_workers=4):
        """Process multiple files in parallel."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.classify_and_preprocess, file) for file in files
            ]

            for future in futures:
                try:
                    future.result()  # raise any exceptions encountered during processing
                except Exception as e:
                    logging.error(f"Error processing file: {e}")
