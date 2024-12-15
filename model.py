from tensorflow.keras import layers, Model


def build_crnn(input_shape, num_classes):
    """
    Build the CRNN model for OCR tasks.

    Args:
        input_shape (tuple): Shape of input images (height, width, channels).
        num_classes (int): Number of unique characters in the dataset.
    
    Returns:
        keras.Model: Compiled CRNN model.
    """
    # Input layer
    inputs = layers.Input(shape=input_shape, name="image_input")

    # Convolutional layers for feature extraction
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    print(f"After MAX1: {x.shape}")

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    print(f"After MAX2: {x.shape}")

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    print(f"After MAX3: {x.shape}")

    # Reshape for RNN layers
    # Collapse height and feature maps for sequence modeling
    x = layers.Reshape(target_shape=(x.shape[2], x.shape[1] * x.shape[3]))(x)   # Combine height and depth
    print(f"After Reshape: {x.shape}")

    # Recurrent layers for sequence modeling
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    print(f"After Bi1: {x.shape}")
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    print(f"After Bi2: {x.shape}")

    # Fully connected layer for classification
    outputs = layers.Dense(num_classes, activation="softmax", name="dense_output")(x)
    print(f"final output: {outputs.shape}")
    # Define and return the model
    model = Model(inputs, outputs, name="CRNN")
    return model
