import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from utils import extract_path_and_filename, extract_formtype_from_path
from data import box_dict, circle_dict

# todo
# 1. experiment by form type (statistics): bounding box (x,y,h,w) histogram / how many detected by form
# 2. is it better to automatic detect or using fixed positon? : update ImageProcessor class
# 3. if save: save cropped bounding boxes else draw boxes on original image

# Set up logging
logging.basicConfig(level=logging.INFO)


class BoxDetector:
    def __init__(self, file_path, output_dir=None):
        self.dir_path, self.file_name = extract_path_and_filename(file_path)
        self.formtype = extract_formtype_from_path(file_path)
        # self.img = cv2.imread(os.path.join(file_dir, file), cv2.IMREAD_GRAYSCALE)
        self.img = cv2.imread(file_path)
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.output_dir = self.dir_path if output_dir is None else output_dir

        if self.img is None:
            logging.error(
                f"Failed to load image from {file_path}. Check the file path or file integrity."
            )
            raise FileNotFoundError(f"Cannot open image: {file_path}")

    @property
    def shape(self):
        return self.img.shape

    def threshold(self, type_flag, threshold=130, value=255, block_size=9, C=5):
        """
        Apply different types of thresholding on the grayscale image.

        :param type_flag: The type of thresholding ('threshold', 'otsu', 'adaptive')
        :param threshold: Threshold value for binary thresholding (ignored for Otsu/adaptive)
        :param value: The value to set for pixels above the threshold
        :param block_size: Block size for adaptive thresholding
        :param C: Constant subtracted from the mean in adaptive thresholding
        :return: Binary image after thresholding
        """
        if type_flag == "threshold":
            _, img_binary = cv2.threshold(
                self.gray_img, threshold, value, cv2.THRESH_BINARY
            )
        elif type_flag == "otsu":
            img_blur = cv2.GaussianBlur(self.gray_img, (5, 5), 0)
            ret, img_binary = cv2.threshold(
                img_blur, -1, value, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )
            logging.info(f"Otsu threshold: {ret}")
        elif type_flag == "adaptive":
            img_binary = cv2.adaptiveThreshold(
                self.gray_img,
                value,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                C,
            )
        else:
            logging.error(f"Unknown thresholding type_flag: {type_flag}")
            raise ValueError(f"Unsupported threshold type: {type_flag}")
        logging.info(f"Image thresholding method {type_flag} applied.")
        return img_binary

    def detect(
        self,
        n_box: int,
        min_box_area: int,
        thresholding=("otsu", 130),
        contour=(cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS),
    ):
        """
        Detect rectangular boxes in the image.

        :param n_box: Number of boxes to detect
        :param min_box_area: Minimum area for a box to be considered
        :param thresholding: Tuple with type of thresholding ('threshold', 'otsu', 'adaptive') and threshold value
        :param contour: Tuple specifying the contour retrieval mode and approximation method
        :return: Dictionary with positions of detected boxes

        contour[0] exmaples:
        cv2.RETR_EXTERNAL: 컨투어 라인 중 가장 바깥쪽의 라인만 찾음
        cv2.RETR_LIST: 모든 컨투어 라인을 찾지만, 상하구조(hierachy)관계를 구성하지 않음
        cv2.RETR_CCOMP: 모든 컨투어 라인을 찾고, 상하구조는 2 단계로 구성함
        cv2.RETR_TREE: 모든 컨투어 라인을 찾고, 모든 상하구조를 구성함

        contour[1] exmaples:
        cv2.CHAIN_APPROX_NONE: 모든 컨투어 포인트를 반환
        cv2.CHAIN_APPROX_SIMPLE: 컨투어 라인을 그릴 수 있는 포인트만 반환
        cv2.CHAIN_APPROX_TC89_L1: Teh_Chin 연결 근사 알고리즘 L1 버전을 적용하여 컨투어 포인트를 줄임
        cv2.CHAIN_APPROX_TC89_KCOS: Teh_Chin 연결 근사 알고리즘 KCOS 버전을 적용하여 컨투어 포인트를 줄임
        """
        type_flag, threshold = thresholding[0], thresholding[1]
        img_binary = self.threshold(type_flag, threshold)

        form_filters = box_dict["filter"]
        # Select the appropriate filter function based on the form type
        selected_filter = form_filters.get(
            self.formtype, lambda x, y: x > 150 and 200 < y < 3200
        )

        # Find contours
        cnts, hierarchy = cv2.findContours(img_binary, contour[0], contour[1])
        logging.info(f"Found {len(cnts)} contours.")

        # Calculate bounding rectangles once and store them for further operations
        bounding_rects = [cv2.boundingRect(c) for c in cnts]

        # Filter contours by area and form-specific criteria
        filtered_cnts = [
            (c, rect)
            for c, rect in zip(cnts, bounding_rects)
            if cv2.contourArea(c) > min_box_area and selected_filter(rect[0], rect[1])
        ]

        # Sort contours by area and limit the number to `n_box`
        filtered_cnts = sorted(
            filtered_cnts, key=lambda x: cv2.contourArea(x[0]), reverse=True
        )[:n_box]
        logging.info(
            f"Filtered and sorted {len(cnts)} contours based on area and y position."
        )

        # Nested box removal
        non_nested_cnts = []
        for i, (c, rect) in enumerate(filtered_cnts):
            x, y, w, h = rect
            is_nested = False
            for j, (c2, rect2) in enumerate(filtered_cnts):
                if i != j:
                    x2, y2, w2, h2 = rect2
                    if (
                        x > x2
                        and y > y2
                        and (x + w) < (x2 + w2)
                        and (y + h) < (y2 + h2)
                    ):
                        is_nested = True
                        break
            if not is_nested:
                non_nested_cnts.append((c, rect))

        form_y_sorters = box_dict["sort"]
        # Select the appropriate y sorter based on the form type
        selected_y = form_y_sorters.get(self.formtype, None)

        # Function to check if contour belongs to a specific group
        def in_group(y, group):
            low, high = group
            if low is None:
                return y < high
            if high is None:
                return y >= low
            return low <= y < high

        if selected_y is None:
            # If no specific y-sorting is defined, sort contours top-left to bottom-right
            # Sort by y then x
            sorted_cnts = sorted(non_nested_cnts, key=lambda x: (x[1][1], x[1][0]))
            logging.info(f"Sorted contours from left-top to right-bottom.")
        else:
            # Sort based on y-group and x position within each group
            sorted_cnts = []
            for group in selected_y:
                group_cnts = [
                    item for item in non_nested_cnts if in_group(item[1][1], group)
                ]
                sorted_cnts.extend(
                    sorted(group_cnts, key=lambda x: x[1][0])
                )  # Sort by x within the group
                logging.info(f"Sorted contours into groups based on y and x positions.")

        # Dictionary to store box positions
        positions = {}
        original_copy = self.img.copy()

        for i, (c, rect) in enumerate(sorted_cnts, 1):
            # Calculate the perimeter
            peri = cv2.arcLength(c, True)

            # Approximate the contour to a polygon
            vertices = cv2.approxPolyDP(c, 0.02 * peri, True)

            # We're looking for rectangular boxes (4 vertices)
            if len(vertices) >= 4:
                x, y, w, h = rect
                # todo: save cropped image

                # Draw rectangle on the original image
                cv2.rectangle(original_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Draw index of the box near the top-left corner of the rectangle
                # putText(image to draw on, box index, position of the text, font, font scale, text color, thickness, line type)
                cv2.putText(
                    original_copy,
                    str(i),
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                # Save position (x, y, w, h)
                positions[i] = (x, y, w, h)
                logging.info(f"Saved box {i} at position {positions[i]}.")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Save the image with bounding boxes drawn
        cv2.imwrite(
            os.path.join(self.output_dir, f"{self.file_name[6:]}_box.png"),
            original_copy,
        )
        logging.info("Saved image with bounding boxes.")

        return positions


class CircleDetector(BoxDetector):

    def __init__(
        self, file_path, blur_size=(3, 3), min_black_pixel=100, output_dir=None
    ):
        super().__init__(file_path, output_dir)
        self.th = min_black_pixel
        if blur_size is not None:
            self.gray_img = cv2.GaussianBlur(self.gray_img, blur_size, 0)

    def detect(
        self, minDist=20, param1=50, param2=30, minRadius=25, maxRadius=40, margin=5
    ):
        circles = cv2.HoughCircles(
            self.gray_img,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius,
        )

        # no circles detected
        if circles is None:
            logging.info("No circle detected.")
            return None

        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :].astype("int"))

        # Define lambda helper for filter
        def in_y_bounds(y, bounds):
            return (bounds[0] is None or y >= bounds[0]) and (
                bounds[1] is None or y < bounds[1]
            )

        # Define form-specific lambda functions for boundingRect filtering
        form_filters = circle_dict["filter"]
        form_y_sorters = circle_dict["sort"]

        selected_filter = form_filters.get(self.formtype, lambda x, y: 200 < y < 3200)
        selected_y = form_y_sorters.get(self.formtype, None)

        filtered_circles = [(x, y, r) for (x, y, r) in circles if selected_filter(x, y)]

        if not filtered_circles:
            logging.info("No circles passed the filter.")
            return None

        # Sorting logic
        if selected_y:
            sorted_circles = []
            for group in selected_y:
                group_circles = [
                    circle
                    for circle in filtered_circles
                    if in_y_bounds(circle[1], group)
                ]
                sorted_circles.extend(
                    sorted(group_circles, key=lambda circle: circle[0])
                )
            logging.info("Sorted circles based on y-group and x position.")
        else:
            sorted_circles = sorted(
                filtered_circles, key=lambda circle: (circle[1], circle[0])
            )
            logging.info("Sorted circles from top-left to bottom-right.")

        # Create a copy of the original image only if we have circles to draw
        original_copy = self.img.copy()
        positions = {}

        # loop over the (x, y) coordinates and radius of the circles
        for i, (x, y, r) in enumerate(sorted_circles, 1):
            # positions[i] = (position, response)
            positions[i] = {"position": (x, y, r), "black": 0, "response": 0}
            # cv2.circle(img array, center, radius, color, thickness)
            cv2.circle(original_copy, (x, y), r, (0, 255, 0), 2)
            cv2.putText(
                original_copy,
                str(i),
                (x - 30, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # Ensure ROI boundaries are valid
            roi_margin = max(0, r - margin)
            roi = self.gray_img[
                max(0, y - roi_margin) : y + roi_margin,
                max(0, x - roi_margin) : x + roi_margin,
            ]
            _, binary = cv2.threshold(roi, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # count the number of 'black' pixels
            pixels = np.sum(binary == 0)
            positions[i]["black"] = pixels
            if pixels >= self.th:
                positions[i]["response"] = 1  # response = 1 (checked)

            logging.info(f"Processed circle {i} with position {positions[i]}.")
        # Save the image with detected circles
        cv2.imwrite(
            os.path.join(self.output_dir, f"{self.file_name[6:]}_circle.png"),
            original_copy,
        )
        logging.info("Saved image with detected circles.")
        return positions


def compare_thresholding(file_path, output_dir="/app/image/output"):
    detector = BoxDetector(file_path, output_dir)

    # Apply different thresholding techniques
    th1 = detector.threshold(type_flag="threshold")
    th2 = detector.threshold(type_flag="otsu")
    th3 = detector.threshold(type_flag="adaptive")

    # Dictionary of images to display
    img_dict = {
        "cv2.threshold": th1,
        "Otsu's Thresholding": th2,
        "Adaptive Thresholding": th3,
    }

    # Plot the thresholded images side by side
    plt.figure(figsize=(10, 4), dpi=100)
    for i, (k, v) in enumerate(img_dict.items()):
        plt.subplot(1, 3, i + 1)
        plt.title(k)
        plt.imshow(v, cmap="gray")
        plt.xticks([])  # Hide x-ticks
        plt.yticks([])  # Hide y-ticks
    plt.tight_layout()
    # Create the output directory if it doesn't exist
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the plot to a file instead of displaying it
        output_path = os.path.join(
            output_dir, f"{detector.file_name}_threshold_comparison.png"
        )
        plt.savefig(output_path)
        plt.close()  # Close the figure to free up memory
        logging.info(f"Threshold comparison saved at {output_path}")

    else:
        plt.show()
