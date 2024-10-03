import os, shutil
import fnmatch
import logging
import cv2
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)


def create_folders(folder_names, parent_dir):
    """Create directories if they don't exist."""
    for folder_name in folder_names:
        folder_path = os.path.join(parent_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)


def get_image_list(root_dir, pattern=None, recursive=False):
    """Get all image files in root_dir with their full paths"""
    file_list = []

    # Walk through the directory
    for dirpath, foldernames, filenames in os.walk(root_dir, topdown=True):
        for filename in filenames:
            extension = os.path.splitext(filename)[1].lower()
            if extension in (".png", ".jpg", ".jpeg"):
                # Check if the file matches the pattern (if a pattern is provided)
                if pattern is None or fnmatch.fnmatch(filename, pattern):
                    # Construct full path to the file
                    file_list.append(os.path.join(dirpath, filename))
        if not recursive:
            break  # If not recursive, stop after the first level
    return file_list


def extract_path_and_filename(file_path):
    """Extract the directory path and file name (without extension) from the given file path."""
    # Extract the directory path
    dir_path = os.path.dirname(file_path)
    # Extract the filename without the extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    return dir_path, file_name


def check_formtype_from_path(file_path, formtype="TS01"):
    """Check if formtype is present as a directory or file name in the given file path."""
    return formtype in file_path.split(os.sep)


def extract_formtype_from_path(file_path, pattern="TS*"):
    folders = os.path.normpath(file_path).split(os.sep)
    found = next(
        (folder for folder in folders if fnmatch.fnmatch(folder, pattern)), None
    )
    return found


def move_to_folder(current_path, new_folder_path, file):
    """Move the image file to the correct folder."""
    new_file_path = os.path.join(new_folder_path, file)

    # Check if the destination directory exists, create it if not
    if not os.path.exists(new_folder_path):
        logging.info(f"Creating destination directory: {new_folder_path}")
        os.makedirs(new_folder_path)

    logging.info(f"Moving {current_path} to {new_file_path}")
    shutil.move(current_path, new_file_path)


def save_image(image, directory, filename):
    cv2.imwrite(os.path.join(directory, filename), image)


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
