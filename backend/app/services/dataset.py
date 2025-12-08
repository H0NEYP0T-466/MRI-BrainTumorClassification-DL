"""Dataset processing pipeline for training and testing data."""
import cv2
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from app.logging_config import logger
from app.config import (
    TRAIN_RAW_TUMOR, TRAIN_RAW_NO_TUMOR,
    TEST_RAW_TUMOR, TEST_RAW_NO_TUMOR,
    TRAIN_PROCESSED_TUMOR, TRAIN_PROCESSED_NO_TUMOR,
    TEST_PROCESSED_TUMOR, TEST_PROCESSED_NO_TUMOR,
    IMAGE_SIZE
)
from app.services.preprocessing import preprocess_image
from app.services.segmentation import segment_brain
from app.utils.files import get_image_files


def process_dataset(force_reprocess: bool = False) -> None:
    """
    Process entire dataset: preprocessing + segmentation.
    
    Iterates through raw dataset folders, applies preprocessing and segmentation,
    and saves to processedDATASETS with mirrored structure.
    
    Args:
        force_reprocess: If True, reprocess even if processed files exist
    """
    logger.info("=" * 60)
    logger.info("Starting dataset processing pipeline")
    logger.info("=" * 60)
    
    # Define raw and processed paths
    dataset_mapping = [
        (TRAIN_RAW_TUMOR, TRAIN_PROCESSED_TUMOR, "Training - Tumor"),
        (TRAIN_RAW_NO_TUMOR, TRAIN_PROCESSED_NO_TUMOR, "Training - No Tumor"),
        (TEST_RAW_TUMOR, TEST_PROCESSED_TUMOR, "Testing - Tumor"),
        (TEST_RAW_NO_TUMOR, TEST_PROCESSED_NO_TUMOR, "Testing - No Tumor"),
    ]
    
    total_processed = 0
    
    for raw_path, processed_path, label in dataset_mapping:
        logger.info(f"\nProcessing: {label}")
        logger.info(f"Source: {raw_path}")
        logger.info(f"Destination: {processed_path}")
        
        # Check if raw path exists
        if not raw_path.exists():
            logger.warning(f"Raw dataset path does not exist: {raw_path}")
            logger.warning(f"Skipping {label}. Please ensure dataset is available.")
            continue
        
        # Get all image files
        image_files = get_image_files(raw_path)
        logger.info(f"Found {len(image_files)} images in {label}")
        
        if len(image_files) == 0:
            logger.warning(f"No images found in {raw_path}")
            continue
        
        # Process each image
        processed_path.mkdir(parents=True, exist_ok=True)
        
        for img_file in tqdm(image_files, desc=f"Processing {label}"):
            output_file = processed_path / img_file.name
            
            # Skip if already processed (unless force_reprocess)
            if output_file.exists() and not force_reprocess:
                continue
            
            try:
                # Preprocess image
                preprocessed, _ = preprocess_image(img_file)
                
                # Segment brain
                segmented, _ = segment_brain(preprocessed)
                
                # Save processed image
                cv2.imwrite(str(output_file), segmented)
                total_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
                continue
    
    logger.info("=" * 60)
    logger.info(f"Dataset processing completed. Total processed: {total_processed}")
    logger.info("=" * 60)


class BrainTumorDataset(Dataset):
    """PyTorch Dataset for brain tumor classification."""
    
    def __init__(
        self,
        data_dirs: List[Tuple[Path, int]],
        transform=None,
        use_processed: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_dirs: List of (directory_path, label) tuples
            transform: Optional transforms to apply
            use_processed: Use processed dataset (default True)
        """
        self.data = []
        self.transform = transform
        
        # Collect all image paths and labels
        for data_dir, label in data_dirs:
            if not data_dir.exists():
                logger.warning(f"Dataset directory does not exist: {data_dir}")
                continue
            
            image_files = get_image_files(data_dir)
            for img_path in image_files:
                self.data.append((img_path, label))
        
        logger.info(f"Dataset initialized with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_train_transforms():
    """Get training data transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_test_transforms():
    """Get test data transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_datasets() -> Tuple[BrainTumorDataset, BrainTumorDataset]:
    """
    Create training and testing datasets.
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    logger.info("Creating training dataset...")
    train_dataset = BrainTumorDataset(
        data_dirs=[
            (TRAIN_PROCESSED_NO_TUMOR, 0),
            (TRAIN_PROCESSED_TUMOR, 1),
        ],
        transform=get_train_transforms()
    )
    
    logger.info("Creating testing dataset...")
    test_dataset = BrainTumorDataset(
        data_dirs=[
            (TEST_PROCESSED_NO_TUMOR, 0),
            (TEST_PROCESSED_TUMOR, 1),
        ],
        transform=get_test_transforms()
    )
    
    return train_dataset, test_dataset
