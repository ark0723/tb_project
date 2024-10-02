import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

# reference
# https://datascienceschool.net/03%20machine%20learning/03.02.03%20%EC%9D%B4%EB%AF%B8%EC%A7%80%20%EC%BB%A8%ED%88%AC%EC%96%B4.html
# https://rahites.tistory.com/55


# todo
# 1. experiment by form type (statistics): bounding box (x,y,h,w) histogram / how many detected by form
# 2. is it better to detect or using fixed positon
# 3. test: deskew bounding box

# Set up logging
logging.basicConfig(level=logging.INFO)


class BoxDetector:
    def __init__(self, file_dir, file, output_dir="/app/image/output"):
        self.file_dir = file_dir
        self.name = file.split(".")[0]
        # self.img = cv2.imread(os.path.join(file_dir, file), cv2.IMREAD_GRAYSCALE)
        self.img = cv2.imread(os.path.join(file_dir, file))
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.output_dir = output_dir

        if self.img is None:
            logging.error(
                f"Failed to load image from {os.path.join(file_dir, file)}. Check the file path or file integrity."
            )
            raise FileNotFoundError(
                f"Cannot open image: {os.path.join(file_dir, file)}"
            )

    @property
    def shape(self):
        return self.img.shape

    def threshold(self, type_flag, threshold=130, value=255, block_size=9, C=5):
        if type_flag == "threshold":
            _, img_binary = cv2.threshold(
                self.gray_img, threshold, value, cv2.THRESH_BINARY
            )
        elif type_flag == "otsu":
            img_blur = cv2.GaussianBlur(self.gray_img, (3, 3), 0)
            ret, img_binary = cv2.threshold(
                img_blur, -1, value, cv2.THRESH_BINARY | cv2.THRESH_OTSU
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

    def detect_boxes(
        self,
        n_box: int,
        min_box_area: int,
        thresholding=("otsu", 130),
        contour=(cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS),
    ):
        """
        mode exmaples:
        cv2.RETR_EXTERNAL: 컨투어 라인 중 가장 바깥쪽의 라인만 찾음
        cv2.RETR_LIST: 모든 컨투어 라인을 찾지만, 상하구조(hierachy)관계를 구성하지 않음
        cv2.RETR_CCOMP: 모든 컨투어 라인을 찾고, 상하구조는 2 단계로 구성함
        cv2.RETR_TREE: 모든 컨투어 라인을 찾고, 모든 상하구조를 구성함

        approx exmaples:
        cv2.CHAIN_APPROX_NONE: 모든 컨투어 포인트를 반환
        cv2.CHAIN_APPROX_SIMPLE: 컨투어 라인을 그릴 수 있는 포인트만 반환
        cv2.CHAIN_APPROX_TC89_L1: Teh_Chin 연결 근사 알고리즘 L1 버전을 적용하여 컨투어 포인트를 줄임
        cv2.CHAIN_APPROX_TC89_KCOS: Teh_Chin 연결 근사 알고리즘 KCOS 버전을 적용하여 컨투어 포인트를 줄임
        """
        type_flag, threshold = thresholding[0], thresholding[1]
        img_binary = self.threshold(type_flag, threshold)

        # Find contours
        cnts, hierarchy = cv2.findContours(img_binary, contour[0], contour[1])
        logging.info(f"Found {len(cnts)} contours.")

        # Sort contours by area and filter based on both min_box_area and
        # y position (200 < y < 3200) to exclude apritag contours
        cnts = sorted(
            [
                c
                for c in cnts
                if cv2.contourArea(c) > min_box_area
                and 200 < cv2.boundingRect(c)[1] < 3200
            ],
            key=cv2.contourArea,
            reverse=True,
        )[1 : n_box + 5]
        logging.info(
            f"Filtered and sorted {len(cnts)} contours based on area and y position."
        )

        # Remove nested bounding boxes
        non_nested_cnts = []
        for i, c in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(c)
            is_nested = False
            for j, c2 in enumerate(cnts):
                if i != j:
                    x2, y2, w2, h2 = cv2.boundingRect(c2)
                    # Check if (x, y, w, h) is inside (x2, y2, w2, h2)
                    if (
                        x > x2
                        and y > y2
                        and (x + w) < (x2 + w2)
                        and (y + h) < (y2 + h2)
                    ):
                        is_nested = True
                        logging.info(
                            f"Contour {i} is nested inside contour {j}, skipping."
                        )
                        break
            if not is_nested:
                non_nested_cnts.append(c)

        # sort conours from left-top to right-bottom
        sorted_cnts = sorted(
            non_nested_cnts,
            key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]),
        )

        # Dictionary to store box positions
        positions = {}
        original_copy = self.img.copy()

        for i, c in enumerate(sorted_cnts, 1):
            # Calculate the perimeter
            peri = cv2.arcLength(c, True)

            # Approximate the contour to a polygon
            vertices = cv2.approxPolyDP(c, 0.01 * peri, True)

            # We're looking for rectangular boxes (4 vertices)
            if len(vertices) == 4:
                x, y, w, h = cv2.boundingRect(c)

                # Draw rectangle on the original image
                cv2.rectangle(
                    original_copy, (x, y), (x + w, y + h), (0, 0, 255), 2
                )  # Red line around boxes

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

        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # Save the image with bounding boxes drawn
            cv2.imwrite(
                os.path.join(self.output_dir, f"{self.name}_box.png"), original_copy
            )
            logging.info("Saved image with bounding boxes.")

        return positions


def compare_thresholding(file_dir, file, output_dir=None):
    detector = BoxDetector(file_dir, file, output_dir)

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
            output_dir, f"{detector.name}_threshold_comparison.png"
        )
        plt.savefig(output_path)
        plt.close()  # Close the figure to free up memory
        logging.info(f"Threshold comparison saved at {output_path}")

    else:
        plt.show()
