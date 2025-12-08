"""Transform utilities for image preprocessing."""
import numpy as np
import cv2
from typing import Tuple

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image

def standardize_intensity(image: np.ndarray) -> np.ndarray:
    """Standardize image intensity (zero mean, unit variance)."""
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        image = (image - mean) / std
    return image

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if it's RGB."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
