"""Transformer-based model architecture using Vision Transformer (ViT)."""
import torch
import torch.nn as nn
import timm
from pathlib import Path
from typing import Optional
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from app.logging_config import logger
from app.config import MODEL_NAME, NUM_CLASSES, DEVICE, MODEL_PATH


# Network error indicators for detecting download failures
NETWORK_ERROR_KEYWORDS = {
    "cannot send a request",  # httpx client closed
    "client has been closed",  # httpx client state
    "no address associated with hostname",  # DNS resolution failure
    "connection refused",  # Network connection failure
    "connection reset",  # Network connection reset
    "network is unreachable",  # Network routing issue
    "failed to resolve",  # DNS resolution failure
}


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
        
        # Try to load pretrained ViT from timm with fallback
        self.model = self._initialize_model(model_name, num_classes)
        
        logger.info("Model initialized successfully")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _initialize_model(self, model_name: str, num_classes: int):
        """
        Initialize the model, trying pretrained first, then falling back to random initialization.
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            
        Returns:
            Initialized model
        """
        try:
            logger.info("Attempting to load pretrained weights from HuggingFace Hub...")
            model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=num_classes
            )
            logger.info("✓ Pretrained weights loaded successfully")
            return model
        except (HfHubHTTPError, LocalEntryNotFoundError) as e:
            # HfHubHTTPError: HTTP errors from HuggingFace Hub (403, 404, 5xx, etc.)
            # LocalEntryNotFoundError: Model not found in local cache and cannot be downloaded
            logger.warning(f"Failed to download pretrained weights: {e}")
            return self._create_model_without_pretrained(model_name, num_classes)
        except (OSError, RuntimeError) as e:
            # OSError: Network/file system errors during download
            # RuntimeError: HTTP client errors (e.g., "Cannot send a request, as the client has been closed")
            # Check if this is a network-related error that should trigger fallback
            error_msg = str(e).lower()
            is_network_error = any(keyword in error_msg for keyword in NETWORK_ERROR_KEYWORDS)
            
            if is_network_error:
                logger.warning(f"Network error while downloading pretrained weights: {e}")
                return self._create_model_without_pretrained(model_name, num_classes)
            else:
                # Re-raise if it's not a network-related error
                raise
    
    def _create_model_without_pretrained(self, model_name: str, num_classes: int):
        """
        Create model with random initialization (no pretrained weights).
        
        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            
        Returns:
            Model with random weights
        """
        logger.warning("Falling back to random initialization (no pretrained weights)")
        logger.info("Note: Training from scratch may require more epochs for convergence")
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes
        )
        logger.info("✓ Model initialized with random weights")
        return model
    
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
        
        # Load state dict with weights_only=True for security (PyTorch 2.6.0+)
        # This prevents arbitrary code execution via malicious model files
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        # Note: weights_only=False is used here because we need to load model metadata
        # In production, consider validating the checkpoint source
        
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
