import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import build_crnn
from data_loader import split_dataset, TFDataLoader
from text_processor import TextProcessor
from config import (
    DATASET_PATH,
    BATCH_SIZE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_TEXT_LENGTH,
    EPOCHS,
    LEARNING_RATE,
)

# Cross-Entropy Loss Function
def cross_entropy_loss(y_true, y_pred):
    """
    Compute Cross-Entropy loss for a batch.

    Args:
        y_true (Tensor): Ground truth labels, padded to the same length.
        y_pred (Tensor): Logits output from the CRNN model.

    Returns:
        Tensor: Mean Cross-Entropy loss across the batch.
    """
    mask = tf.not_equal(y_true, 0)
    max_time_steps = tf.shape(y_pred)[1]
    y_pred_padded = tf.pad(y_pred, [[0, 0], [0, MAX_TEXT_LENGTH - max_time_steps], [0, 0]])
    mask_flat = tf.reshape(mask, [-1])
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred_padded, [-1, tf.shape(y_pred)[-1]])
    y_true_flat_masked = tf.boolean_mask(y_true_flat, mask_flat)
    y_pred_flat_masked = tf.boolean_mask(y_pred_flat, mask_flat)
    num_classes = tf.shape(y_pred)[-1]
    y_true_one_hot = tf.one_hot(y_true_flat_masked, depth=num_classes)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true_one_hot, logits=y_pred_flat_masked)
    return tf.reduce_mean(loss)


# Accuracy Metric
def compute_accuracy(y_true, y_pred):
    """
    Computes the character-level accuracy.

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted logits from the model.

    Returns:
        Tensor: Accuracy score as a Tensor.
    """
    # Decode predictions (get the index of the max probability)
    y_pred_decoded = tf.argmax(y_pred, axis=-1)  # Shape: (batch_size, MAX_TEXT_LENGTH)

    # Truncate or pad predictions to match the ground truth shape
    max_label_length = tf.shape(y_true)[1]
    y_pred_decoded = y_pred_decoded[:, :max_label_length]  # Truncate predictions
    y_pred_decoded = tf.pad(  # Pad predictions if they are shorter than y_true
        y_pred_decoded,
        [[0, 0], [0, max_label_length - tf.shape(y_pred_decoded)[1]]],
        constant_values=0,
    )

    # Create a mask to ignore padding (label == 0 is padding)
    mask = tf.not_equal(y_true, 0)

    # Convert both y_true and y_pred_decoded to the same type
    y_true = tf.cast(y_true, tf.int32)
    y_pred_decoded = tf.cast(y_pred_decoded, tf.int32)

    # Compare masked ground truth and predictions
    correct = tf.equal(tf.boolean_mask(y_true, mask), tf.boolean_mask(y_pred_decoded, mask))

    # Compute accuracy as the mean of correct predictions
    return tf.reduce_mean(tf.cast(correct, tf.float32))





# Edit Distance Metric
def compute_edit_distance(y_true, y_pred):
    """
    Computes the average edit distance.

    Args:
        y_true (Tensor): Ground truth labels of shape (batch_size, max_label_length).
        y_pred (Tensor): Predicted logits from the model of shape (batch_size, time_steps, num_classes).

    Returns:
        Tensor: Average edit distance as a Tensor.
    """
    # Decode predictions to get class indices
    y_pred_decoded = tf.argmax(y_pred, axis=-1)  # Shape: (batch_size, time_steps)

    # Ensure y_true is padded to max_label_length
    max_label_length = tf.shape(y_true)[1]
    y_pred_decoded = y_pred_decoded[:, :max_label_length]  # Truncate predictions if needed
    y_pred_decoded = tf.pad(
        y_pred_decoded,
        [[0, 0], [0, max_label_length - tf.shape(y_pred_decoded)[1]]],  # Pad if shorter
        constant_values=0,
    )

    # Convert dense labels (y_true and y_pred_decoded) to sparse tensors
    y_true_sparse = tf.sparse.from_dense(tf.cast(y_true, tf.int32))
    y_pred_sparse = tf.sparse.from_dense(tf.cast(y_pred_decoded, tf.int32))

    # Compute edit distances
    edit_distances = tf.edit_distance(y_pred_sparse, y_true_sparse, normalize=True)

    # Return the average edit distance
    return tf.reduce_mean(edit_distances)





# Preprocess a batch
def preprocess_batch(images, labels, text_processor):
    """
    Preprocesses a batch of images and labels for training.

    Args:
        images (list): List of image tensors.
        labels (list): List of corresponding ground truth labels as strings.
        text_processor (TextProcessor): TextProcessor instance for label encoding.

    Returns:
        tuple: Preprocessed images and padded label sequences as Tensors.
    """
    label_sequences = [text_processor.text_to_sequence(label) for label in labels if label]
    label_padded = tf.keras.preprocessing.sequence.pad_sequences(
        label_sequences, maxlen=MAX_TEXT_LENGTH, padding="post", value=0
    )
    images = tf.convert_to_tensor(images)
    if len(images.shape) == 3:
        images = tf.expand_dims(images, axis=-1)
    return images, label_padded


# Custom Training Step
@tf.function
def custom_train_step(model, optimizer, images, labels):
    """
    Performs a single training step.

    Args:
        model (tf.keras.Model): CRNN model to train.
        optimizer (tf.keras.optimizers.Optimizer): Optimizer for updating model weights.
        images (Tensor): Batch of input images.
        labels (Tensor): Corresponding ground truth labels.

    Returns:
        tuple: Loss, accuracy, and edit distance for the batch.
    """
    with tf.GradientTape() as tape:
        y_pred = model(images, training=True)
        loss = cross_entropy_loss(labels, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Compute metrics
    accuracy = compute_accuracy(labels, y_pred)
    edit_distance = compute_edit_distance(labels, y_pred)

    return loss, accuracy, edit_distance


# Training Loop
def train():
    """
    Main training loop for the CRNN model.
    Splits the dataset, initializes the model and optimizer, and trains the model
    while logging training and validation metrics.
    """
    (train_paths, train_labels), (val_paths, val_labels), _ = split_dataset(DATASET_PATH, test_size=0.2, val_size=0.1)
    CHARACTERS = "".join(sorted(set("".join(train_labels))))
    text_processor = TextProcessor(CHARACTERS)
    num_classes = len(CHARACTERS) + 1

    train_tf_dataset = TFDataLoader.create_tf_dataset(train_paths, train_labels, BATCH_SIZE, text_processor)
    val_tf_dataset = TFDataLoader.create_tf_dataset(val_paths, val_labels, BATCH_SIZE, text_processor)

    crnn = build_crnn((IMAGE_HEIGHT, IMAGE_WIDTH, 1), num_classes)
    optimizer = Adam(LEARNING_RATE)

    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        epoch_loss, epoch_accuracy, epoch_edit_distance = 0.0, 0.0, 0.0
        train_batches = 0

        for batch, (images, labels) in enumerate(train_tf_dataset):
            loss, accuracy, edit_distance = custom_train_step(crnn, optimizer, images, labels)
            epoch_loss += loss.numpy()
            epoch_accuracy += accuracy.numpy()
            epoch_edit_distance += edit_distance.numpy()
            train_batches += 1
            print(
                f"Batch {batch + 1}, Loss: {loss.numpy():.4f}, Accuracy: {accuracy.numpy():.4f}, Edit Distance: {edit_distance.numpy():.4f}"
            )

        print(f"Epoch {epoch + 1} Training Metrics -> Loss: {epoch_loss / train_batches:.4f}, Accuracy: {epoch_accuracy / train_batches:.4f}, Edit Distance: {epoch_edit_distance / train_batches:.4f}")

        print("Validating...")
        val_loss, val_accuracy, val_edit_distance = 0.0, 0.0, 0.0
        val_batches = 0
        for images, labels in val_tf_dataset:
            y_pred = crnn(images, training=False)
            val_loss += cross_entropy_loss(labels, y_pred).numpy()
            val_accuracy += compute_accuracy(labels, y_pred).numpy()
            val_edit_distance += compute_edit_distance(labels, y_pred).numpy()
            val_batches += 1
        print(
            f"Validation Metrics -> Loss: {val_loss / val_batches:.4f}, Accuracy: {val_accuracy / val_batches:.4f}, Edit Distance: {val_edit_distance / val_batches:.4f}"
        )

    crnn.save("crnn_ocr_model_cross_entropy.keras")
    print("Model saved to crnn_ocr_model_cross_entropy.keras")


if __name__ == "__main__":
    train()
