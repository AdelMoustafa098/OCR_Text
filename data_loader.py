import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from utils import preprocess_image
from config import *

def split_dataset(dataset_path, test_size=0.2, val_size=0.1):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        dataset_path (str): Path to the dataset.
        test_size (float): Fraction of data for testing.
        val_size (float): Fraction of data for validation.

    Returns:
        tuple: Lists of file paths and corresponding labels for training, validation, and test sets.
    """
    image_paths = []
    labels = []

    # Iterate over dataset directory to collect image paths and labels
    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".jpg"):
            image_paths.append(os.path.join(dataset_path, file_name))
            label_file = file_name.replace(".jpg", ".txt")
            label_path = os.path.join(dataset_path, label_file)
            if os.path.exists(label_path):
                with open(label_path, "r", encoding="utf-8") as f:
                    labels.append(f.read().strip())
            else:
                print(file_name)
                labels.append(None)  # Handle unlabeled images

    # Split into train and temp (validation + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=(test_size + val_size), random_state=42
    )

    # Further split temp into validation and test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=(test_size / (test_size + val_size)),
        random_state=42,
    )

    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
    )


class OCRDataset(Sequence):
    def __init__(self, image_paths, labels, batch_size):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_image_paths = self.image_paths[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_labels = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]

        images = [
            preprocess_image(img_path, (IMAGE_HEIGHT, IMAGE_WIDTH))
            for img_path in batch_image_paths
        ]

        images = np.array(images)

        # Drop batches with fewer samples than the batch size
        if len(images) < self.batch_size:
            return None

        return images, batch_labels

    def on_epoch_end(self):
        temp = list(zip(self.image_paths, self.labels))
        np.random.shuffle(temp)
        self.image_paths, self.labels = zip(*temp)


class TFDataLoader:
    @staticmethod
    def create_tf_dataset(image_paths, labels, batch_size, text_processor):
        """
        Creates a TensorFlow dataset from the given image paths and labels.

        Args:
            image_paths (list): List of image file paths.
            labels (list): List of corresponding text labels.
            batch_size (int): Number of samples per batch.
            text_processor (TextProcessor): Instance to preprocess text labels.

        Returns:
            tf.data.Dataset: A batched and shuffled TensorFlow dataset.
        """
        def generator():
            for img_path, label in zip(image_paths, labels):
                image = preprocess_image(img_path, (IMAGE_HEIGHT, IMAGE_WIDTH))
                label_sequence = text_processor.text_to_sequence(label)  # Convert text to sequence
                yield image, label_sequence

        output_signature = (
            tf.TensorSpec(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),  # Labels are sequences of integers
        )

        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

        # Shuffle, batch, and prefetch
        dataset = dataset.shuffle(buffer_size=1000).padded_batch(
            batch_size,
            padded_shapes=(
                (IMAGE_HEIGHT, IMAGE_WIDTH, 1),  # Image shape
                (MAX_TEXT_LENGTH,),  # Pad labels to MAX_TEXT_LENGTH
            ),
            drop_remainder=True,
        ).prefetch(tf.data.AUTOTUNE)

        return dataset
