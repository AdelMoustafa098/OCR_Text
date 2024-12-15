import pytest
import tensorflow as tf
from train import ctc_loss, custom_train_step, preprocess_batch
from model import build_crnn
from data_loader import OCRDataset
from text_processor import TextProcessor


@pytest.fixture
def setup_environment():
    """
    Set up the training environment for tests, including the dataset, text processor, and model.
    """
    # Characters for TextProcessor (Arabic letters, Arabic digits, and English digits)
    characters = (
        "ءاأإآبتثجحخدذرزسشصضطظعغفقكلمنهوي"
        "٠١٢٣٤٥٦٧٨٩"
        "0123456789"
    )
    text_processor = TextProcessor(characters)

    # Input shape and model
    input_shape = (80, 500, 1)
    num_classes = len(characters) + 1  # Adding blank token
    model = build_crnn(input_shape, num_classes)

    # Create a mock dataset
    images = tf.random.uniform((10, 80, 500, 1), minval=0, maxval=1, dtype=tf.float32)
    labels = ["مرحبا", "عالم", "تعلم", "ذكاء", "اصطناعي", "بيانات", "برمجة", "حاسوب", "رقم", "١٢٣"]

    # Define a function to preprocess mock images and labels without relying on file paths
    def mock_dataset_generator():
        for image, label in zip(images, labels):
            yield image, label

    dataset = tf.data.Dataset.from_generator(
        mock_dataset_generator,
        output_signature=(
            tf.TensorSpec(shape=(80, 500, 1), dtype=tf.float32),  # Image shape
            tf.TensorSpec(shape=(), dtype=tf.string),             # Label as string
        )
    )

    return text_processor, model, dataset.batch(2)


def test_preprocess_batch(setup_environment):
    """
    Test the preprocess_batch function.
    """
    text_processor, _, dataset = setup_environment

    for images, labels in dataset:
        # Preprocess the batch
        images, padded_labels, input_lengths, label_lengths = preprocess_batch(images, labels, text_processor)

        # Assert shapes and types
        assert images.shape == (2, 80, 500, 1)  # Batch size of 2
        assert padded_labels.shape[1] <= 32     # MAX_TEXT_LENGTH
        assert len(input_lengths) == 2
        assert len(label_lengths) == 2
        break