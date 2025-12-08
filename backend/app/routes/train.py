"""Training endpoint."""
from fastapi import APIRouter, HTTPException
from app.schemas import TrainRequest, TrainResponse
from app.logging_config import logger
from app.services.training import train_model
from app.services.dataset import process_dataset
from app.config import MODEL_PATH
import asyncio

router = APIRouter()

# Track training status
training_status = {"is_training": False}


@router.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest = TrainRequest()):
    """
    Train the model on processed dataset.
    
    This endpoint:
    1. Processes the dataset (preprocessing + segmentation) if needed
    2. Trains the model with verbose logging
    3. Saves the best model
    4. Returns training summary
    """
    logger.info("Training request received")
    logger.info(f"Parameters - epochs: {request.epochs}, batch_size: {request.batch_size}")
    
    # Check if already training
    if training_status["is_training"]:
        logger.warning("Training already in progress")
        raise HTTPException(
            status_code=409,
            detail="Training already in progress. Please wait for it to complete."
        )
    
    try:
        training_status["is_training"] = True
        
        # Process dataset first
        logger.info("Processing dataset before training...")
        await asyncio.to_thread(process_dataset, force_reprocess=False)
        
        # Train model
        logger.info("Starting model training...")
        metrics = await asyncio.to_thread(
            train_model,
            epochs=request.epochs,
            batch_size=request.batch_size
        )
        
        logger.info("Training completed successfully")
        
        return TrainResponse(
            epochs=request.epochs,
            final_metrics={
                'val_acc': metrics['best_val_acc'] / 100.0,  # Convert to 0-1 range
                'train_acc': metrics['final_train_acc'] / 100.0,
                'best_epoch': metrics['best_epoch'],
                'training_time_minutes': metrics['training_time_minutes']
            },
            model_path=str(MODEL_PATH)
        )
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error during training: {str(e)}"
        )
    finally:
        training_status["is_training"] = False
