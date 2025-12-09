"""Prediction endpoint."""
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas import PredictionResponse, PreprocessingSteps
from app.logging_config import logger
from app.services.inference import predict_image

router = APIRouter()

# Global model reference (will be set by main.py)
current_model = {"model": None}


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict brain tumor classification from uploaded MRI image.
    
    Accepts: PNG, JPG, JPEG image files
    Returns: Predicted class (tumor/no_tumor) and confidence score
    """
    logger.info(f"Prediction request received for file: {file.filename}")
    
    # Check if model is loaded
    if current_model["model"] is None:
        logger.error("Model not loaded - cannot make prediction")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first or ensure model file exists."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a PNG or JPG image."
        )
    
    try:
        # Read file content
        logger.info("Reading uploaded file...")
        contents = await file.read()
        
        # Run prediction with preprocessing steps
        logger.info("Starting prediction pipeline...")
        class_name, confidence, preprocessing_steps = predict_image(
            current_model["model"], 
            contents, 
            return_preprocessing_steps=True
        )
        
        logger.info(f"Prediction completed: class={class_name}, confidence={confidence:.4f}")
        
        return PredictionResponse(
            class_name=class_name,
            confidence=confidence,
            preprocessing_steps=preprocessing_steps
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )
