"""Utility functions 
"""
import cv2
import numpy as np

def load_image(image_path):
    """
    Load an image and preprocess it (resize, grayscale, normalize).
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        np.ndarray: Preprocessed image.
    """
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to fixed dimensions
    image = cv2.resize(image, (500, 80))
    
    # Normalize pixel values (scale to 0-1)
    image = image / 255.0
    
    # Expand dimensions to add channel for model input
    image = np.expand_dims(image, axis=-1)
    
    return image
