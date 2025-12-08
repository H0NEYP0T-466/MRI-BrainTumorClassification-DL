"""Image preprocessing with Digital Image Processing techniques."""
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Union, Tuple
import torch

from app.logging_config import logger
from app.config import IMAGE_SIZE


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Apply Non-Local Means Denoising to remove noise."""
    return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)


def sharpen_image(image: np.ndarray) -> np.ndarray:
    """Sharpen image using unsharp masking."""
    # Create Gaussian blur
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    # Unsharp mask
    sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return sharpened


def enhance_edges(image: np.ndarray) -> np.ndarray:
    """Enhance edges using morphological gradient."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    # Blend with original
    enhanced = cv2.addWeighted(image, 0.8, gradient, 0.2, 0)
    return enhanced


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """Normalize image intensity to [0, 255] range."""
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return (image * 255).astype(np.uint8)


def preprocess_image(
    image_path: Union[str, Path, Image.Image],
    save_path: Union[str, Path] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply comprehensive DIP preprocessing pipeline to enhance MRI images.
    
    Steps:
    1. Load and convert to grayscale
    2. Denoise (Non-Local Means)
    3. Contrast enhancement (CLAHE)
    4. Edge sharpening
    5. Edge enhancement
    6. Intensity normalization
    7. Resize to target size
    
    Args:
        image_path: Path to image or PIL Image
        save_path: Optional path to save preprocessed image
        
    Returns:
        Tuple of (preprocessed_image, resized_for_model)
    """
    logger.info(f"Preprocessing started for: {image_path}")
    
    # Load image
    if isinstance(image_path, (str, Path)):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
    elif isinstance(image_path, Image.Image):
        # Convert PIL Image to numpy array
        image = np.array(image_path.convert('L'))
    else:
        raise ValueError(f"Unsupported image type: {type(image_path)}")
    
    logger.info("Applying denoising...")
    # Step 1: Denoise
    denoised = denoise_image(image)
    
    logger.info("Applying contrast enhancement (CLAHE)...")
    # Step 2: Contrast enhancement
    enhanced = apply_clahe(denoised)
    
    logger.info("Applying sharpening...")
    # Step 3: Sharpen
    sharpened = sharpen_image(enhanced)
    
    logger.info("Enhancing edges...")
    # Step 4: Edge enhancement
    edge_enhanced = enhance_edges(sharpened)
    
    logger.info("Normalizing intensity...")
    # Step 5: Normalize intensity
    normalized = normalize_intensity(edge_enhanced)
    
    # Save preprocessed image if path provided
    if save_path:
        cv2.imwrite(str(save_path), normalized)
        logger.info(f"Preprocessed image saved to: {save_path}")
    
    logger.info("Resizing for model input...")
    # Resize for model
    resized = cv2.resize(normalized, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    
    logger.info("Preprocessing completed successfully")
    return normalized, resized


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert preprocessed image to PyTorch tensor.
    
    Args:
        image: Preprocessed image (grayscale, uint8)
        
    Returns:
        Tensor of shape (1, 3, H, W) normalized to [0, 1]
    """
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Convert grayscale to 3-channel (for ViT compatibility)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=0)
    else:
        image = np.transpose(image, (2, 0, 1))
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(image).unsqueeze(0)
    
    return tensor
