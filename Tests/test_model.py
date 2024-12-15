import pytest
import numpy as np
from model import build_crnn

@pytest.fixture
def setup_model():
    """
    Fixture to initialize the CRNN model with mock input shape and character classes.
    """
    input_shape = (80, 500, 1)  # Height, Width, Channels
    num_classes = 75  # Arabic + English digits + blank token
    model = build_crnn(input_shape, num_classes)
    return model, input_shape, num_classes


def test_model_input_output_dimensions(setup_model):
    """
    Test the CRNN model for correct input and output dimensions.
    """
    model, input_shape, num_classes = setup_model

    # Mock input batch of 2 images
    batch_size = 2
    mock_input = np.random.rand(batch_size, *input_shape).astype(np.float32)

    # Pass the mock input through the model
    output = model(mock_input)

    # Expected temporal dimension after convolutional downsampling
    downsampling_factor = 8  # From three (2, 2) MaxPooling layers
    time_steps = input_shape[1] // downsampling_factor

    # Verify the output shape
    assert output.shape == (batch_size, time_steps, num_classes), (
        f"Expected output shape (batch_size, time_steps, num_classes) = "
        f"({batch_size}, {time_steps}, {num_classes}), but got {output.shape}"
    )
