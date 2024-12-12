from data_loader import OCRDataset
from config import DATASET_PATH, BATCH_SIZE

if __name__ == "__main__":
    # Initialize dataset loader
    ocr_dataset = OCRDataset(DATASET_PATH, BATCH_SIZE)

    # Example: Access the first batch
    images, labels = ocr_dataset[0]

    print("Loaded a batch of images and labels:")
    print(f"Images shape: {images.shape}")
    print(f"Sample labels: {labels[:5]}")
