"""Pydantic schemas for API request/response models."""
from pydantic import BaseModel, Field
from typing import Dict, Optional

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    class_name: str = Field(..., alias="class", description="Predicted class: tumor or no_tumor")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    
    class Config:
        populate_by_name = True

class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")

class TrainRequest(BaseModel):
    """Request model for training endpoint."""
    epochs: int = Field(default=10, ge=1, le=100, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=1, le=128, description="Batch size for training")

class TrainResponse(BaseModel):
    """Response model for training endpoint."""
    epochs: int = Field(..., description="Number of epochs trained")
    final_metrics: Dict[str, float] = Field(..., description="Final training metrics")
    model_path: str = Field(..., description="Path where model was saved")
