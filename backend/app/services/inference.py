"""Inference service for model predictions."""
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Union
import io

from app.logging_config import logger
from app.config import DEVICE, CLASS_NAMES
from app.services.preprocessing import preprocess_image
from app.services.segmentation import segment_brain
from app.services.dataset import get_test_transforms


def predict_image(
    model: torch.nn.Module,
    image_input: Union[str, Path, Image.Image, bytes]
) -> Tuple[str, float]:
    """
    Predict class and confidence for a single image.
    
    Pipeline:
    1. Load/convert image
    2. Preprocess (DIP techniques)
    3. Segment brain
    4. Transform for model input
    5. Run inference
    6. Apply softmax for confidence
    
    Args:
        model: Trained model
        image_input: Image file path, PIL Image, or bytes
        
    Returns:
        Tuple of (class_name, confidence)
    """
    logger.info("Starting prediction pipeline...")
    
    # Step 1: Load image
    if isinstance(image_input, bytes):
        logger.info("Loading image from bytes...")
        image = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, (str, Path)):
        logger.info(f"Loading image from path: {image_input}")
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        logger.info("Using provided PIL Image")
        image = image_input
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Step 2 & 3: Preprocess and segment
    logger.info("Preprocessing and segmenting image...")
    # Convert PIL to numpy for preprocessing
    image_np = np.array(image.convert('L'))
    
    # Preprocess
    from app.services.preprocessing import (
        denoise_image, apply_clahe, sharpen_image,
        enhance_edges, normalize_intensity
    )
    
    denoised = denoise_image(image_np)
    enhanced = apply_clahe(denoised)
    sharpened = sharpen_image(enhanced)
    edge_enhanced = enhance_edges(sharpened)
    preprocessed = normalize_intensity(edge_enhanced)
    
    # Segment
    segmented, _ = segment_brain(preprocessed)
    
    # Convert back to PIL RGB for transforms
    segmented_rgb = Image.fromarray(segmented).convert('RGB')
    
    # Step 4: Transform for model
    logger.info("Applying model transforms...")
    transform = get_test_transforms()
    image_tensor = transform(segmented_rgb).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(DEVICE)
    
    # Step 5: Run inference
    logger.info("Running model inference...")
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # Step 6: Apply softmax for confidence
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
        confidence = confidence.item()
        predicted_class = predicted_class.item()
    
    class_name = CLASS_NAMES[predicted_class]
    
    logger.info(f"Prediction completed: class={class_name}, confidence={confidence:.4f}")
    
    return class_name, confidence


def batch_predict(
    model: torch.nn.Module,
    images: list
) -> list:
    """
    Run batch prediction on multiple images.
    
    Args:
        model: Trained model
        images: List of image inputs
        
    Returns:
        List of (class_name, confidence) tuples
    """
    logger.info(f"Starting batch prediction for {len(images)} images...")
    
    results = []
    for idx, image in enumerate(images):
        logger.info(f"Processing image {idx + 1}/{len(images)}")
        try:
            class_name, confidence = predict_image(model, image)
            results.append((class_name, confidence))
        except Exception as e:
            logger.error(f"Error processing image {idx + 1}: {e}")
            results.append(("error", 0.0))
    
    logger.info("Batch prediction completed")
    return results
