"""Transformer-based model architecture using Vision Transformer (ViT)."""
import torch
import torch.nn as nn
import timm
from pathlib import Path
from typing import Optional

from app.logging_config import logger
from app.config import MODEL_NAME, NUM_CLASSES, DEVICE, MODEL_PATH


class BrainTumorClassifier(nn.Module):
    """
    Vision Transformer-based classifier for brain tumor detection.
    
    Uses timm's pretrained ViT (vit_base_patch16_224) as backbone,
    with custom classification head for binary classification.
    """
    
    def __init__(self, model_name: str = MODEL_NAME, num_classes: int = NUM_CLASSES):
        super(BrainTumorClassifier, self).__init__()
        
        logger.info(f"Initializing model: {model_name}")
        logger.info(f"Number of classes: {num_classes}")
        
        # Load pretrained ViT from timm
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes
        )
        
        logger.info("Model initialized successfully")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        return self.model(x)


def create_model() -> BrainTumorClassifier:
    """
    Create a new model instance.
    
    Returns:
        BrainTumorClassifier instance
    """
    logger.info("Creating new model instance...")
    model = BrainTumorClassifier()
    model = model.to(DEVICE)
    logger.info(f"Model moved to device: {DEVICE}")
    return model


def save_model(model: nn.Module, path: Path = MODEL_PATH) -> None:
    """
    Save model state dict to file.
    
    Args:
        model: Model to save
        path: Path to save model
    """
    logger.info(f"Saving model to: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': MODEL_NAME,
        'num_classes': NUM_CLASSES,
    }, path)
    
    logger.info("Model saved successfully")


def load_model(path: Path = MODEL_PATH) -> Optional[BrainTumorClassifier]:
    """
    Load model from saved state dict.
    
    Args:
        path: Path to model file
        
    Returns:
        Loaded model or None if load fails
    """
    logger.info(f"Attempting to load model from: {path}")
    
    if not path.exists():
        logger.warning(f"Model file not found at: {path}")
        return None
    
    try:
        # Create model instance
        model = create_model()
        
        # Load state dict
        checkpoint = torch.load(path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        
        logger.info("Model loaded successfully")
        logger.info(f"Model architecture: {checkpoint.get('model_name', 'Unknown')}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about the model.
    
    Args:
        model: Model instance
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_name': MODEL_NAME,
        'num_classes': NUM_CLASSES,
        'device': str(DEVICE)
    }
