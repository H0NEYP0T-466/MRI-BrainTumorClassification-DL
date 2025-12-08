"""Configuration settings for the application."""
import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
PROCESSED_DATASET_DIR = DATASET_DIR / "processedDATASETS"
MODEL_DIR = BASE_DIR / "Model"
MODEL_PATH = MODEL_DIR / "model.pt"

# Dataset paths - Windows compatible
WINDOWS_DATASET_PATH = Path("X:/file/FAST_API/MRI-BrainTumorClassification-DL/backend/dataset")
if WINDOWS_DATASET_PATH.exists():
    DATASET_DIR = WINDOWS_DATASET_PATH
    PROCESSED_DATASET_DIR = DATASET_DIR / "processedDATASETS"

# Training paths
TRAIN_RAW_TUMOR = DATASET_DIR / "Training" / "tumor"
TRAIN_RAW_NO_TUMOR = DATASET_DIR / "Training" / "no_tumor"
TEST_RAW_TUMOR = DATASET_DIR / "Testing" / "tumor"
TEST_RAW_NO_TUMOR = DATASET_DIR / "Testing" / "no_tumor"

# Processed paths
TRAIN_PROCESSED_TUMOR = PROCESSED_DATASET_DIR / "training" / "tumor"
TRAIN_PROCESSED_NO_TUMOR = PROCESSED_DATASET_DIR / "training" / "no_tumor"
TEST_PROCESSED_TUMOR = PROCESSED_DATASET_DIR / "testing" / "tumor"
TEST_PROCESSED_NO_TUMOR = PROCESSED_DATASET_DIR / "testing" / "no_tumor"

# Model configuration
IMAGE_SIZE = 224  # Standard ViT input size
NUM_CLASSES = 2
CLASS_NAMES = ["no_tumor", "tumor"]

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training configuration
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Model name (timm)
MODEL_NAME = "vit_base_patch16_224"  # Latest ViT architecture

# Create necessary directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_PROCESSED_TUMOR.mkdir(parents=True, exist_ok=True)
TRAIN_PROCESSED_NO_TUMOR.mkdir(parents=True, exist_ok=True)
TEST_PROCESSED_TUMOR.mkdir(parents=True, exist_ok=True)
TEST_PROCESSED_NO_TUMOR.mkdir(parents=True, exist_ok=True)
