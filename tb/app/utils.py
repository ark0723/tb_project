import os
import logging
import shutil
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)


def create_folders(folder_names, parent_dir):
    """Create directories if they don't exist."""
    for folder_name in folder_names:
        folder_path = os.path.join(parent_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)


def save_image(image, directory, filename):
    cv2.imwrite(os.path.join(directory, filename), image)


def move_to_folder(current_path, new_folder_path, file):
    """Move the image file to the correct folder."""
    new_file_path = os.path.join(new_folder_path, file)

    # Check if the destination directory exists, create it if not
    if not os.path.exists(new_folder_path):
        logging.info(f"Creating destination directory: {new_folder_path}")
        os.makedirs(new_folder_path)

    logging.info(f"Moving {current_path} to {new_file_path}")
    shutil.move(current_path, new_file_path)
