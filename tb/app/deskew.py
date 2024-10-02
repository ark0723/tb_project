import cv2
import numpy as np


def deskew_image(image_path, output_path=None):
    # Load the image in grayscale mode (ensure single-channel)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        raise FileNotFoundError(f"Cannot load image from {image_path}")

    # Apply binary thresholding with Otsu's method
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the coordinates of non-zero pixels (where the binary image is white)
    coords = np.column_stack(np.where(binary > 0))

    # Get the angle of the minimum area rectangle enclosing those points
    angle = cv2.minAreaRect(coords)[-1]

    # Adjust the angle for proper deskewing
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Get the image size and calculate the center
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)

    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the affine transformation (rotate the image)
    rotated = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    # Save the output image if the path is provided
    if output_path:
        cv2.imwrite(output_path, rotated)

    # Return the deskewed image
    return rotated
