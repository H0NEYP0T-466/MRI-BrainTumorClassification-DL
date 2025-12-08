# MRI Brain Tumor Classification - Backend

Deep learning-based brain tumor detection system using Vision Transformer (ViT) architecture.

## Features

- **Transformer-based Model**: Uses timm's pretrained Vision Transformer (vit_base_patch16_224)
- **Digital Image Processing**: Advanced preprocessing with CLAHE, denoising, sharpening, edge enhancement
- **Brain Segmentation**: Automated skull-stripping using OpenCV morphological operations
- **Comprehensive Logging**: Structured logging with logger.info() for all operations
- **RESTful API**: FastAPI endpoints for prediction, training, and health checks
- **Confidence Scores**: Softmax-based confidence scores for predictions

## Architecture

### Model
- **Base**: Vision Transformer (ViT) - `vit_base_patch16_224`
- **Pretrained**: Yes (ImageNet)
- **Fine-tuned**: Binary classification (tumor/no_tumor)
- **Input Size**: 224x224 RGB images

### Preprocessing Pipeline
1. Denoising (Non-Local Means)
2. Contrast Enhancement (CLAHE)
3. Edge Sharpening
4. Edge Enhancement (Morphological Gradient)
5. Intensity Normalization
6. Resize to 224x224

### Segmentation
- Otsu's thresholding
- Morphological operations (opening/closing)
- Largest connected component extraction
- Brain mask refinement

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare dataset structure:
```
backend/dataset/
├── Training/
│   ├── tumor/
│   └── no_tumor/
└── Testing/
    ├── tumor/
    └── no_tumor/
```

3. Process dataset (optional, done automatically during training):
```python
from app.services.dataset import process_dataset
process_dataset()
```

## Usage

### Start Server

```bash
# From backend directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or:

```bash
python -m app.main
```

Server will be available at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### API Endpoints

#### 1. Health Check
```bash
GET /health

Response:
{
  "status": "ok",
  "model_loaded": true
}
```

#### 2. Predict
```bash
POST /predict
Content-Type: multipart/form-data

Form Data:
- file: <image file>

Response:
{
  "class": "tumor",
  "confidence": 0.935
}
```

Example with curl:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@mri_scan.jpg"
```

#### 3. Train Model
```bash
POST /train
Content-Type: application/json

Body:
{
  "epochs": 10,
  "batch_size": 32
}

Response:
{
  "epochs": 10,
  "final_metrics": {
    "val_acc": 0.91,
    "train_acc": 0.95,
    "best_epoch": 8,
    "training_time_minutes": 45.2
  },
  "model_path": "backend/Model/model.pt"
}
```

## Configuration

Edit `app/config.py` to customize:

- `IMAGE_SIZE`: Input image size (default: 224)
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Number of training epochs (default: 10)
- `LEARNING_RATE`: Learning rate (default: 1e-4)
- `MODEL_NAME`: timm model name (default: "vit_base_patch16_224")

## Dataset Paths

The system supports both relative and Windows absolute paths:

- **Default**: `backend/dataset/`
- **Windows Alternative**: `X:/file/FAST_API/MRI-BrainTumorClassification-DL/backend/dataset/`

Processed datasets are stored in:
- `backend/dataset/processedDATASETS/`

## Model Storage

Trained models are saved to:
- `backend/Model/model.pt`

The model file contains:
- Model state dict
- Model architecture name
- Number of classes

## Logging

All operations are logged with structured logging:

```
2024-12-08 13:30:00 - app - INFO - Server starting...
2024-12-08 13:30:01 - app - INFO - Model loaded successfully
2024-12-08 13:30:05 - app - INFO - Prediction request received for file: scan.jpg
2024-12-08 13:30:06 - app - INFO - Preprocessing started...
2024-12-08 13:30:07 - app - INFO - Segmentation completed
2024-12-08 13:30:08 - app - INFO - Inference completed: class=tumor, confidence=0.923
```

## Training Workflow

1. Place raw images in `dataset/Training/` and `dataset/Testing/` folders
2. Start server (model will attempt to load existing model)
3. Call `POST /train` to process dataset and train model
4. Monitor training progress in server logs
5. Best model automatically saved to `Model/model.pt`
6. Use `POST /predict` for inference

## Development

### Project Structure
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── logging_config.py    # Logging setup
│   ├── schemas.py           # Pydantic models
│   ├── routes/
│   │   ├── health.py        # Health check endpoint
│   │   ├── predict.py       # Prediction endpoint
│   │   └── train.py         # Training endpoint
│   ├── services/
│   │   ├── preprocessing.py # Image preprocessing
│   │   ├── segmentation.py  # Brain segmentation
│   │   ├── dataset.py       # Dataset handling
│   │   ├── model.py         # Model architecture
│   │   ├── training.py      # Training pipeline
│   │   └── inference.py     # Inference logic
│   └── utils/
│       ├── files.py         # File utilities
│       └── transforms.py    # Transform utilities
├── Model/                   # Saved models
├── dataset/                 # Raw and processed datasets
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Performance

- Training time: ~30-60 minutes on GPU (depends on dataset size)
- Inference time: ~1-2 seconds per image
- Model size: ~350 MB

## Requirements

See `requirements.txt` for complete list. Key dependencies:

- fastapi
- uvicorn
- torch
- torchvision
- timm (for ViT models)
- opencv-python
- scikit-image
- pillow
- numpy

## Troubleshooting

### Model not loading
- Ensure `backend/Model/model.pt` exists
- Check file permissions
- Train model using `POST /train`

### Dataset not found
- Check dataset paths in `config.py`
- Ensure folder structure matches requirements
- Check Windows path if using absolute path

### Out of memory during training
- Reduce `BATCH_SIZE` in config.py
- Reduce `IMAGE_SIZE` (not recommended, may affect accuracy)
- Use CPU if GPU memory insufficient (slower)

### Import errors
- Install all dependencies: `pip install -r requirements.txt`
- Use Python 3.8 or higher
- Consider using virtual environment

## License

MIT License

## Authors

H0NEYP0T-466
