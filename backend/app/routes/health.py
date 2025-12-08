"""Health check endpoint."""
from fastapi import APIRouter
from app.schemas import HealthResponse
from app.logging_config import logger

router = APIRouter()

# Global state for model loaded status
model_loaded_state = {"loaded": False}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns server status and model loaded state.
    """
    logger.info("Health check requested")
    
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded_state["loaded"]
    )
