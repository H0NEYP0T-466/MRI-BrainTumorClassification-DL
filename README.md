# MRI Brain Tumor Classification - Deep Learning System

A complete end-to-end deep learning system for brain tumor detection from MRI scans, featuring a React + TypeScript frontend and FastAPI backend with Vision Transformer (ViT) architecture.

![Dark Theme](https://img.shields.io/badge/theme-dark-111111)
![React](https://img.shields.io/badge/React-19.2.0-61DAFB?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.9.3-3178C6?logo=typescript)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch)

## Features

### 🧠 Deep Learning Model
- **Transformer-based Architecture**: Vision Transformer (ViT) via timm
- **Pretrained on ImageNet**: Fine-tuned for medical imaging
- **Binary Classification**: Tumor vs No Tumor detection
- **Confidence Scores**: Softmax-based probability estimates

### 🎨 Modern UI
- **Dark Theme**: Base background #111 with contrasting accent colors
- **Single Page Application**: All features in one view
- **Drag-and-Drop Upload**: Intuitive file upload interface
- **Real-time Progress**: Animated progress indicators
- **Results Visualization**: Clear presentation of predictions and confidence scores

### 🔬 Advanced Image Processing
- **Digital Image Processing Pipeline**:
  - Non-Local Means Denoising
  - CLAHE Contrast Enhancement
  - Edge Sharpening
  - Morphological Edge Enhancement
  - Intensity Normalization
- **Brain Segmentation**: Automated skull-stripping using OpenCV
- **Preprocessing**: Standardized pipeline for optimal model input

### 📊 Comprehensive Logging
- Structured logging with `logger.info()` throughout
- Verbose training progress with per-epoch and per-batch logs
- All operations tracked for debugging and monitoring

## Architecture

```
MRI-BrainTumorClassification-DL/
├── frontend/                    # React + TypeScript SPA
│   ├── src/
│   │   ├── components/         # UI components
│   │   ├── api/                # API client
│   │   ├── styles/             # CSS modules
│   │   ├── App.tsx             # Main application
│   │   └── main.tsx            # Entry point
│   ├── package.json
│   └── vite.config.ts
│
└── backend/                     # Python FastAPI server
    ├── app/
    │   ├── routes/             # API endpoints
    │   ├── services/           # ML pipeline
    │   ├── utils/              # Helper functions
    │   ├── main.py             # FastAPI app
    │   ├── config.py           # Configuration
    │   └── schemas.py          # Pydantic models
    ├── Model/                  # Saved models
    ├── dataset/                # Training data
    └── requirements.txt
```

## Quick Start

### Prerequisites

- **Node.js**: 16+ (for frontend)
- **Python**: 3.8+ (for backend)
- **CUDA**: Optional, for GPU acceleration

### Installation

#### 1. Clone Repository
```bash
git clone https://github.com/H0NEYP0T-466/MRI-BrainTumorClassification-DL.git
cd MRI-BrainTumorClassification-DL
```

#### 2. Setup Backend

```bash
cd backend
pip install -r requirements.txt
```

#### 3. Setup Frontend

```bash
# From root directory
npm install
```

### Running the Application

#### Start Backend Server

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

#### Start Frontend Development Server

```bash
# From root directory
npm run dev
```

Frontend will be available at: `http://localhost:5173`

### Dataset Setup

Place your MRI images in the following structure:

```
backend/dataset/
├── Training/
│   ├── tumor/      # MRI scans with tumors
│   └── no_tumor/   # MRI scans without tumors
└── Testing/
    ├── tumor/
    └── no_tumor/
```

Supported formats: PNG, JPG, JPEG

### Training the Model

1. Ensure dataset is in place
2. Call the training endpoint:

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"epochs": 10, "batch_size": 32}'
```

Or use the API documentation at `http://localhost:8000/docs`

The model will be automatically saved to `backend/Model/model.pt`

## API Endpoints

### Health Check
```http
GET /health
```

Returns server status and model loaded state.

### Predict
```http
POST /predict
Content-Type: multipart/form-data

Form Data:
- file: <image file>
```

Returns prediction with class and confidence score.

### Train
```http
POST /train
Content-Type: application/json

Body:
{
  "epochs": 10,
  "batch_size": 32
}
```

Trains the model and returns training metrics.

## Configuration

### Frontend (.env)
```env
VITE_API_BASE_URL=http://localhost:8000
```

### Backend (app/config.py)
```python
IMAGE_SIZE = 224           # Model input size
BATCH_SIZE = 32            # Training batch size
EPOCHS = 10                # Training epochs
LEARNING_RATE = 1e-4       # Learning rate
MODEL_NAME = "vit_base_patch16_224"  # timm model
```

## Technology Stack

### Frontend
- **React** 19.2.0
- **TypeScript** 5.9.3
- **Vite** 7.2.4
- **CSS Modules** for styling

### Backend
- **FastAPI** 0.115.12
- **PyTorch** 2.5.1
- **timm** 1.0.12 (Vision Transformer)
- **OpenCV** 4.10.0
- **scikit-image** 0.24.0

## Color Palette

| Color | Hex | Usage |
|-------|-----|-------|
| Background | `#111` | Primary background |
| Text | `#e0e0e0` | Primary text |
| Info | `#4dd0e1` | Info accents, highlights |
| Danger | `#ff6b6b` | Tumor detection, errors |
| Success | `#00e676` | No tumor, success states |
| Warning | `#ffd93d` | Warnings |

## Performance

- **Training Time**: 30-60 minutes (GPU) for 10 epochs
- **Inference Time**: 1-2 seconds per image
- **Model Size**: ~350 MB
- **Accuracy**: Depends on dataset quality and size

## Development

### Frontend Development
```bash
npm run dev      # Start dev server
npm run build    # Build for production
npm run lint     # Run ESLint
npm run preview  # Preview production build
```

### Backend Development
```bash
# Run with auto-reload
uvicorn app.main:app --reload

# Run tests (if available)
pytest

# Format code
black app/
```

## Troubleshooting

### Model Not Loading
- Ensure `backend/Model/model.pt` exists
- Train model using `/train` endpoint
- Check file permissions

### Dataset Not Found
- Verify dataset path in `backend/app/config.py`
- Ensure folder structure matches requirements
- Check for Windows path if using absolute path

### Connection Refused
- Verify backend is running on port 8000
- Check CORS settings in `backend/app/main.py`
- Update `VITE_API_BASE_URL` in frontend `.env`

### Out of Memory
- Reduce `BATCH_SIZE` in config
- Use CPU if GPU memory insufficient
- Close other applications

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Authors

- **H0NEYP0T-466**

## Acknowledgments

- Vision Transformer (ViT) by Google Research
- timm library by Ross Wightman
- FastAPI framework
- React and Vite communities

## Citation

If you use this project in your research, please cite:

```bibtex
@software{mri_tumor_classification,
  title={MRI Brain Tumor Classification System},
  author={H0NEYP0T-466},
  year={2024},
  url={https://github.com/H0NEYP0T-466/MRI-BrainTumorClassification-DL}
}
```

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Note**: This is a research/educational project. For clinical use, proper validation, regulatory approval, and medical expert supervision are required.
