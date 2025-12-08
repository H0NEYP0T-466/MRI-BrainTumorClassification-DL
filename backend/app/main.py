"""Main FastAPI application for MRI Brain Tumor Classification."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.logging_config import logger
from app.config import MODEL_PATH
from app.services.model import load_model
from app.routes import health, predict, train


# Global model reference
model_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown."""
    # Startup
    logger.info("=" * 80)
    logger.info("SERVER STARTING - MRI Brain Tumor Classification")
    logger.info("=" * 80)
    
    # Load model
    logger.info(f"Attempting to load model from: {MODEL_PATH}")
    global model_instance
    model_instance = load_model(MODEL_PATH)
    
    if model_instance is not None:
        logger.info("Model loaded successfully")
        health.model_loaded_state["loaded"] = True
        predict.current_model["model"] = model_instance
    else:
        logger.warning("Model not found or failed to load")
        logger.warning("Server will start but predictions will not be available")
        logger.warning("Please train the model using POST /train endpoint")
        health.model_loaded_state["loaded"] = False
        predict.current_model["model"] = None
    
    logger.info("=" * 80)
    logger.info("SERVER READY")
    logger.info("=" * 80)
    
    yield
    
    # Shutdown
    logger.info("Server shutting down...")


# Create FastAPI app
app = FastAPI(
    title="MRI Brain Tumor Classification API",
    description="Deep learning-based brain tumor detection from MRI scans",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default port
        "http://localhost:3000",  # Alternative React port
        "http://localhost:5174",  # Alternative Vite port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(train.router, tags=["Training"])


@app.get("/")
async def root():
    """Root endpoint."""
    logger.info("Root endpoint accessed")
    return {
        "message": "MRI Brain Tumor Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "train": "/train",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
