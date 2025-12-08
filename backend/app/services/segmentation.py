"""Brain segmentation (skull stripping) using OpenCV."""
import numpy as np
import cv2
from pathlib import Path
from typing import Union

from app.logging_config import logger


def extract_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extract the largest connected component from a binary mask.
    This helps isolate the brain region from artifacts.
    """
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    
    if num_labels <= 1:
        return binary_mask
    
    # Find the largest component (excluding background at index 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create mask with only the largest component
    largest_component = (labels == largest_label).astype(np.uint8) * 255
    
    return largest_component


def segment_brain(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment brain tissue from MRI using skull stripping approximation.
    
    Uses OpenCV-based approach:
    1. Otsu thresholding to separate brain from background
    2. Morphological operations to clean up the mask
    3. Largest connected component extraction
    4. Apply mask to original image
    
    This is a basic approach. For production, consider:
    - MONAI's brain extraction transforms
    - HD-BET (deep learning-based)
    - FSL BET
    
    Args:
        image: Preprocessed grayscale image (uint8)
        
    Returns:
        Tuple of (segmented_image, brain_mask)
    """
    logger.info("Starting brain segmentation...")
    
    # Step 1: Otsu's thresholding
    logger.info("Applying Otsu thresholding...")
    _, binary_mask = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    # Step 2: Morphological operations to clean noise
    logger.info("Applying morphological operations...")
    # Opening: remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # Closing: fill small holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Step 3: Extract largest connected component (brain)
    logger.info("Extracting largest connected component (brain)...")
    brain_mask = extract_largest_component(closed)
    
    # Step 4: Additional morphological refinement
    logger.info("Refining brain mask...")
    # Dilate slightly to ensure we don't lose brain tissue
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    brain_mask = cv2.dilate(brain_mask, kernel_dilate, iterations=1)
    
    # Smooth mask edges
    brain_mask = cv2.GaussianBlur(brain_mask, (5, 5), 0)
    _, brain_mask = cv2.threshold(brain_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Step 5: Apply mask to original image
    logger.info("Applying brain mask to image...")
    segmented_image = cv2.bitwise_and(image, image, mask=brain_mask)
    
    logger.info("Brain segmentation completed successfully")
    
    return segmented_image, brain_mask


def segment_and_save(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    preprocessed_image: np.ndarray = None
) -> np.ndarray:
    """
    Segment brain from image and save result.
    
    Args:
        image_path: Path to original image (for reference)
        output_path: Path to save segmented image
        preprocessed_image: Already preprocessed image array
        
    Returns:
        Segmented image array
    """
    logger.info(f"Segmenting image: {image_path}")
    
    # If preprocessed image not provided, load it
    if preprocessed_image is None:
        preprocessed_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if preprocessed_image is None:
            raise ValueError(f"Could not load image from {image_path}")
    
    # Perform segmentation
    segmented_image, brain_mask = segment_brain(preprocessed_image)
    
    # Save segmented image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), segmented_image)
    logger.info(f"Segmented image saved to: {output_path}")
    
    return segmented_image


# Import Tuple for type hints
from typing import Tuple
