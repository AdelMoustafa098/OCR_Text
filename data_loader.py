"""Loading and preprocessing the dataset
"""

import os
import numpy as np
from utils import load_image
from tensorflow.keras.utils import Sequence

class OCRDataset(Sequence):
    """
    Custom data loader for OCR task.
    """
    def __init__(self, dataset_path, batch_size):
        """
        Initialize the dataset loader.

        Args:
            dataset_path (str): Path to the dataset folder.
            batch_size (int): Number of samples per batch.
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.image_paths, self.labels = self._load_data()

    def _load_data(self):
        """
        Load image paths and corresponding labels from the dataset folder.

        Returns:
            list: List of image paths.
            list: List of corresponding labels.
        """
        image_paths = []
        labels = []
        for file_name in os.listdir(self.dataset_path):
            if file_name.endswith(".jpg"):
                image_paths.append(os.path.join(self.dataset_path, file_name))
                label_file = file_name.replace(".jpg", ".txt")
                label_path = os.path.join(self.dataset_path, label_file)
                if os.path.exists(label_path):
                    with open(label_path, "r", encoding="utf-8") as f:
                        labels.append(f.read().strip())
                else:
                    labels.append(None)  # Handle unlabeled images
        return image_paths, labels

    def __len__(self):
        """
        Get the number of batches per epoch.

        Returns:
            int: Number of batches.
        """
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        """
        Get a batch of data.

        Args:
            idx (int): Batch index.

        Returns:
            tuple: Tuple of (images, labels).
        """
        batch_image_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Load and preprocess images
        images = [load_image(img_path) for img_path in batch_image_paths]
        images = np.array(images)

        return images, batch_labels
