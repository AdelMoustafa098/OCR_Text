"""Utility functions 
"""
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from config import DATASET_PATH

def preprocess_image(image_path, target_size):
    """
    Load and preprocess an image.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired image size (height, width).

    Returns:
        np.array: Preprocessed image.
    """
    img = load_img(image_path, color_mode="grayscale", target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array


def find_max_text_length(dataset_path):
    """
    Find the maximum text length in the dataset.

    Args:
        dataset_path (str): Path to the dataset.

    Returns:
        int: Maximum length of the text in the dataset.
    """
    max_length = 0
    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".txt"):
            with open(
                os.path.join(dataset_path, file_name), "r", encoding="utf-8"
            ) as f:
                text = f.read().strip()
                max_length = max(max_length, len(text))
    return max_length



