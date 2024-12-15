import utils as ut
from config import *
import numpy as np
import os
import cv2

# Mock data: Create a sample grayscale image (100x200)


def test_preprocess_image():
    np.random.seed(42)
    # read image get a random image number dataset is named in number for (0) to (19999)
    img_name = str(np.random.randint(0, 19999))
    img_path = os.path.join(DATASET_PATH, f"{img_name}.jpg")

    # Test the function
    processed_image = ut.load_image(img_path)

    # print(f"Processed image shape: {processed_image.shape}")
    # print(f"Processed image type: {type(processed_image)}")
    # print(f"Pixel value range: {processed_image.min()} to {processed_image.max()}")

    assert processed_image.shape == (80, 500, 1)
    assert type(processed_image) == np.ndarray
    assert processed_image.min() >= 0.0
    assert processed_image.max() <= 1.0


def test_find_max_text_length():
    # create an example dataset
    os.makedirs("mock_dataset", exist_ok=True)

    with open("mock_dataset/sample1.txt", "w", encoding="utf-8") as f:
        # len = 11
        f.write("hello world")
    with open("mock_dataset/sample2.txt", "w", encoding="utf-8") as f:
        # len = 25
        f.write("this is a longer text!!!$")
    with open("mock_dataset/sample3.txt", "w", encoding="utf-8") as f:
        # len = 5
        f.write("short")

    with open("mock_dataset/sample4.txt", "w", encoding="utf-8") as f:
        # len = 24
        f.write("text with trailing space       ")
    
    with open("mock_dataset/sample4.txt", "w", encoding="utf-8") as f:
        # len = 24
        f.write("       text with starting space")

    max_length = ut.find_max_text_length("mock_dataset")
    # print(type(max_length))
    # print(f"Maximum text length in the dataset: {max_length}")

    assert type(max_length) == int
    assert max_length == 25
