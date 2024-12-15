import os
import numpy as np
from config import DATASET_PATH
from data_loader import split_dataset, OCRDataset
import shutil

# Mock constants
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 500
BATCH_SIZE = 2

def setup_mock_dataset():
    # create new folder for testing data
    np.random.seed(42)
    os.makedirs("data_loader_mock_test", exist_ok=True)
    img_name_lst = np.random.randint(0, 19999, size=10)
    img_name_lst = [str(img) + ".jpg" for img in img_name_lst]
    images = []
    labels = []
    files_lst = os.listdir(DATASET_PATH)

    for f in img_name_lst:
        idx = files_lst.index(f)
        img = files_lst[idx]
        images.append(img)
        label_name = f.replace(".jpg", ".txt")
        if label_name in files_lst:
            labels.append(label_name)

    for image, label in zip(images, labels):
        new_img_path = 'data_loader_mock_test/' + image
        new_lbl_path = 'data_loader_mock_test/' + label
        shutil.copy(os.path.join(DATASET_PATH, image), new_img_path)
        shutil.copy(os.path.join(DATASET_PATH, label), new_lbl_path)

def test_split_dataset():
    """
    Test the split_dataset function.
    """
    setup_mock_dataset()
    train, val, test = split_dataset("data_loader_mock_test", test_size=0.2, val_size=0.1)

    # Assertions
    assert len(train[0]) > 0  # Check training images exist
    assert len(val[0]) > 0  # Check validation images exist
    assert len(test[0]) > 0  # Check test images exist

    assert len(train[1]) > 0  # Check training labels exist
    assert len(val[1]) > 0  # Check validation labels exist
    assert len(test[1]) > 0  # Check test labels exist


def test_ocr_dataset():
    """
    Test the OCRDataset class.
    """
    image_paths = []
    labels = []
    for file_name in os.listdir("data_loader_mock_test"):
        if file_name.endswith(".jpg"):
            image_paths.append(os.path.join("./data_loader_mock_test/", file_name))
            label_file = file_name.replace(".jpg", ".txt")
            label_path = os.path.join("./data_loader_mock_test/", label_file)
            if os.path.exists(label_path):
                with open(label_path, "r", encoding="utf-8") as f:
                    labels.append(f.read().strip())
            else:
                labels.append(None)  # Handle unlabeled images


    # Initialize dataset
    dataset = OCRDataset(image_paths, labels, BATCH_SIZE)

    # Check length
    assert len(dataset) == 5  # 10 samples, batch size 2 => 5 full batch

    # Fetch a batch
    images, batch_labels = dataset[0]
    assert images.shape == (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1)  # Check image shape
    assert len(batch_labels) == BATCH_SIZE  # Check labels


if __name__ == "__main__":
    import pytest
    pytest.main(["-v", "test_data_loader.py"])