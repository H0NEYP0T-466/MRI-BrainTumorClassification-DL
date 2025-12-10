# MRI Brain Tumor Classification - Deep Learning System

<p align="center">
  <!-- Core -->
  <img src="https://img.shields.io/github/license/H0NEYP0T-466/MRI-BrainTumorClassification-DL?style=for-the-badge&color=brightgreen" alt="License">
  <img src="https://img.shields.io/github/stars/H0NEYP0T-466/MRI-BrainTumorClassification-DL?style=for-the-badge&color=yellow" alt="Stars">
  <img src="https://img.shields.io/github/forks/H0NEYP0T-466/MRI-BrainTumorClassification-DL?style=for-the-badge&color=blue" alt="Forks">
  <img src="https://img.shields.io/github/issues/H0NEYP0T-466/MRI-BrainTumorClassification-DL?style=for-the-badge&color=red" alt="Issues">
  <img src="https://img.shields.io/github/issues-pr/H0NEYP0T-466/MRI-BrainTumorClassification-DL?style=for-the-badge&color=orange" alt="Pull Requests">
  <img src="https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge" alt="Contributions Welcome">
</p>

<p align="center">
  <!-- Activity -->
  <img src="https://img.shields.io/github/last-commit/H0NEYP0T-466/MRI-BrainTumorClassification-DL?style=for-the-badge&color=purple" alt="Last Commit">
  <img src="https://img.shields.io/github/commit-activity/m/H0NEYP0T-466/MRI-BrainTumorClassification-DL?style=for-the-badge&color=teal" alt="Commit Activity">
  <img src="https://img.shields.io/github/repo-size/H0NEYP0T-466/MRI-BrainTumorClassification-DL?style=for-the-badge&color=blueviolet" alt="Repo Size">
  <img src="https://img.shields.io/github/languages/code-size/H0NEYP0T-466/MRI-BrainTumorClassification-DL?style=for-the-badge&color=indigo" alt="Code Size">
</p>

<p align="center">
  <!-- Languages -->
  <img src="https://img.shields.io/github/languages/top/H0NEYP0T-466/MRI-BrainTumorClassification-DL?style=for-the-badge&color=critical" alt="Top Language">
  <img src="https://img.shields.io/github/languages/count/H0NEYP0T-466/MRI-BrainTumorClassification-DL?style=for-the-badge&color=success" alt="Languages Count">
</p>

<p align="center">
  <!-- Community -->
  <img src="https://img.shields.io/badge/Docs-Available-green?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Documentation">
  <img src="https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red?style=for-the-badge" alt="Open Source Love">
</p>

<p align="center">
  <strong>A complete end-to-end deep learning system for brain tumor detection from MRI scans, featuring a React + TypeScript frontend and FastAPI backend with Vision Transformer (ViT) architecture.</strong>
</p>

---

## 🔗 Quick Links

- [Demo](#-usage-examples) | [Documentation](#-table-of-contents) | [Issues](https://github.com/H0NEYP0T-466/MRI-BrainTumorClassification-DL/issues) | [Contributing](CONTRIBUTING.md)

## 📑 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Dependencies & Packages](#-dependencies--packages)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage Examples](#-usage-examples)
- [API Endpoints](#-api-endpoints)
- [Folder Structure](#-folder-structure)
- [Configuration](#-configuration)
- [Performance](#-performance)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Security](#-security)
- [Code of Conduct](#-code-of-conduct)
- [Support](#-support)
- [Citation](#-citation)

## ✨ Features

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

## 🏗️ Architecture

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

## 🛠 Tech Stack

### Languages
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)

### Frameworks & Libraries

**Frontend:**

![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)

**Backend:**

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

**Machine Learning:**

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

### DevOps / Tools

![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)
![ESLint](https://img.shields.io/badge/ESLint-4B32C3?style=for-the-badge&logo=eslint&logoColor=white)
![npm](https://img.shields.io/badge/npm-CB3837?style=for-the-badge&logo=npm&logoColor=white)
![pip](https://img.shields.io/badge/pip-3775A9?style=for-the-badge&logo=pypi&logoColor=white)

## 📦 Dependencies & Packages

### Runtime Dependencies (Frontend)

<details open>
<summary><strong>Click to expand</strong></summary>

[![react](https://img.shields.io/npm/v/react?style=for-the-badge&label=react&color=61DAFB)](https://www.npmjs.com/package/react) - A JavaScript library for building user interfaces

[![react-dom](https://img.shields.io/npm/v/react-dom?style=for-the-badge&label=react-dom&color=61DAFB)](https://www.npmjs.com/package/react-dom) - Entry point for React DOM

</details>

### Dev Dependencies (Frontend)

<details>
<summary><strong>Click to expand</strong></summary>

[![vite](https://img.shields.io/npm/v/vite?style=for-the-badge&label=vite&color=646CFF)](https://www.npmjs.com/package/vite) - Next generation frontend tooling

[![typescript](https://img.shields.io/npm/v/typescript?style=for-the-badge&label=typescript&color=3178C6)](https://www.npmjs.com/package/typescript) - TypeScript language support

[![eslint](https://img.shields.io/npm/v/eslint?style=for-the-badge&label=eslint&color=4B32C3)](https://www.npmjs.com/package/eslint) - Linting utility for JavaScript/TypeScript

[![@vitejs/plugin-react](https://img.shields.io/npm/v/@vitejs/plugin-react?style=for-the-badge&label=@vitejs/plugin-react&color=646CFF)](https://www.npmjs.com/package/@vitejs/plugin-react) - Official Vite React plugin

[![@types/react](https://img.shields.io/npm/v/@types/react?style=for-the-badge&label=@types/react&color=3178C6)](https://www.npmjs.com/package/@types/react) - TypeScript definitions for React

[![@types/react-dom](https://img.shields.io/npm/v/@types/react-dom?style=for-the-badge&label=@types/react-dom&color=3178C6)](https://www.npmjs.com/package/@types/react-dom) - TypeScript definitions for React DOM

[![@types/node](https://img.shields.io/npm/v/@types/node?style=for-the-badge&label=@types/node&color=3178C6)](https://www.npmjs.com/package/@types/node) - TypeScript definitions for Node.js

[![typescript-eslint](https://img.shields.io/npm/v/typescript-eslint?style=for-the-badge&label=typescript-eslint&color=3178C6)](https://www.npmjs.com/package/typescript-eslint) - TypeScript tooling for ESLint

[![eslint-plugin-react-hooks](https://img.shields.io/npm/v/eslint-plugin-react-hooks?style=for-the-badge&label=eslint-plugin-react-hooks&color=61DAFB)](https://www.npmjs.com/package/eslint-plugin-react-hooks) - ESLint rules for React Hooks

[![eslint-plugin-react-refresh](https://img.shields.io/npm/v/eslint-plugin-react-refresh?style=for-the-badge&label=eslint-plugin-react-refresh&color=61DAFB)](https://www.npmjs.com/package/eslint-plugin-react-refresh) - ESLint plugin for React Fast Refresh

[![@eslint/js](https://img.shields.io/npm/v/@eslint/js?style=for-the-badge&label=@eslint/js&color=4B32C3)](https://www.npmjs.com/package/@eslint/js) - ESLint JavaScript rules

[![globals](https://img.shields.io/npm/v/globals?style=for-the-badge&label=globals&color=00A67E)](https://www.npmjs.com/package/globals) - Global identifiers from JavaScript environments

</details>

### Runtime Dependencies (Backend)

<details open>
<summary><strong>Click to expand</strong></summary>

[![fastapi](https://img.shields.io/pypi/v/fastapi?style=for-the-badge&label=fastapi&color=009688)](https://pypi.org/project/fastapi/) - Modern, fast web framework for building APIs with Python

[![uvicorn](https://img.shields.io/pypi/v/uvicorn?style=for-the-badge&label=uvicorn&color=2094F3)](https://pypi.org/project/uvicorn/) - Lightning-fast ASGI server

[![pydantic](https://img.shields.io/pypi/v/pydantic?style=for-the-badge&label=pydantic&color=E92063)](https://pypi.org/project/pydantic/) - Data validation using Python type hints

[![torch](https://img.shields.io/pypi/v/torch?style=for-the-badge&label=torch&color=EE4C2C)](https://pypi.org/project/torch/) - PyTorch deep learning framework

[![torchvision](https://img.shields.io/pypi/v/torchvision?style=for-the-badge&label=torchvision&color=EE4C2C)](https://pypi.org/project/torchvision/) - Image and video datasets and models for PyTorch

[![timm](https://img.shields.io/pypi/v/timm?style=for-the-badge&label=timm&color=FFA500)](https://pypi.org/project/timm/) - PyTorch image models - Vision Transformer (ViT) implementation

[![numpy](https://img.shields.io/pypi/v/numpy?style=for-the-badge&label=numpy&color=013243)](https://pypi.org/project/numpy/) - Fundamental package for scientific computing

[![pandas](https://img.shields.io/pypi/v/pandas?style=for-the-badge&label=pandas&color=150458)](https://pypi.org/project/pandas/) - Data manipulation and analysis library

[![opencv-python](https://img.shields.io/pypi/v/opencv-python?style=for-the-badge&label=opencv-python&color=5C3EE8)](https://pypi.org/project/opencv-python/) - Computer vision and image processing library

[![scikit-image](https://img.shields.io/pypi/v/scikit-image?style=for-the-badge&label=scikit-image&color=FF6F00)](https://pypi.org/project/scikit-image/) - Image processing in Python

[![scikit-learn](https://img.shields.io/pypi/v/scikit-learn?style=for-the-badge&label=scikit-learn&color=F7931E)](https://pypi.org/project/scikit-learn/) - Machine learning library

[![pillow](https://img.shields.io/pypi/v/pillow?style=for-the-badge&label=pillow&color=3776AB)](https://pypi.org/project/pillow/) - Python Imaging Library (PIL Fork)

[![monai](https://img.shields.io/pypi/v/monai?style=for-the-badge&label=monai&color=00A67E)](https://pypi.org/project/monai/) - Medical Open Network for AI

[![python-multipart](https://img.shields.io/pypi/v/python-multipart?style=for-the-badge&label=python-multipart&color=3776AB)](https://pypi.org/project/python-multipart/) - Multipart form data parser

[![tqdm](https://img.shields.io/pypi/v/tqdm?style=for-the-badge&label=tqdm&color=FFC107)](https://pypi.org/project/tqdm/) - Progress bar for Python

[![aiofiles](https://img.shields.io/pypi/v/aiofiles?style=for-the-badge&label=aiofiles&color=3776AB)](https://pypi.org/project/aiofiles/) - File support for asyncio

</details>

## 🚀 Quick Start

### Prerequisites

- **Node.js**: 16+ (for frontend)
- **Python**: 3.8+ (for backend)
- **CUDA**: Optional, for GPU acceleration

## 💻 Installation

### 1. Clone Repository
```bash
git clone https://github.com/H0NEYP0T-466/MRI-BrainTumorClassification-DL.git
cd MRI-BrainTumorClassification-DL
```

### 2. Setup Backend

```bash
cd backend
pip install -r requirements.txt
```

### 3. Setup Frontend

```bash
# From root directory
npm install
```

## ⚡ Usage Examples

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

### Using the UI

1. **Upload MRI Image**: Drag and drop or click to upload an MRI scan
2. **View Processing Steps**: Watch real-time preprocessing visualization
3. **Get Prediction**: Receive tumor detection results with confidence scores
4. **Analyze Results**: Review detailed prediction information

### Using the API

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Make Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/mri-scan.jpg"
```

Response:
```json
{
  "prediction": "tumor",
  "confidence": 0.95,
  "preprocessing_steps": [...]
}
```

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

#### Option 1: Standalone Training Script (Recommended)

The easiest way to train the model is using the standalone training script with comprehensive logging:

```bash
cd backend
python train.py
```

This will:
- ✅ Check dataset availability
- 🔄 Preprocess images automatically
- 📊 Show detailed dataset statistics
- 🧠 Display model architecture
- 📈 Train with verbose logging (epochs, batches, loss, accuracy)
- 💾 Save the best model automatically

**Custom parameters:**
```bash
# Train for more epochs
python train.py --epochs 20

# Use smaller batch size
python train.py --batch-size 16

# Adjust learning rate
python train.py --lr 0.00005

# View all options
python train.py --help
```

See [backend/TRAINING_GUIDE.md](backend/TRAINING_GUIDE.md) for detailed documentation.

#### Option 2: API Endpoint

Alternatively, you can train via the API endpoint:

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"epochs": 10, "batch_size": 32}'
```

Or use the API documentation at `http://localhost:8000/docs`

The model will be automatically saved to `backend/Model/model.pt`

## 📂 Folder Structure

```
MRI-BrainTumorClassification-DL/
├── .github/                     # GitHub configuration
│   ├── ISSUE_TEMPLATE/         # Issue templates
│   │   ├── bug_report.yml
│   │   ├── feature_request.yml
│   │   └── config.yml
│   └── pull_request_template.md
├── backend/                     # Python FastAPI backend
│   ├── app/
│   │   ├── routes/             # API routes
│   │   │   ├── health.py
│   │   │   ├── predict.py
│   │   │   └── train.py
│   │   ├── services/           # Business logic
│   │   │   ├── dataset.py
│   │   │   ├── inference.py
│   │   │   ├── model.py
│   │   │   ├── preprocessing.py
│   │   │   ├── segmentation.py
│   │   │   └── training.py
│   │   ├── utils/              # Utility functions
│   │   │   ├── files.py
│   │   │   └── transforms.py
│   │   ├── config.py           # Configuration
│   │   ├── logging_config.py   # Logging setup
│   │   ├── main.py             # FastAPI application
│   │   └── schemas.py          # Pydantic models
│   ├── Model/                  # Saved model weights
│   │   └── README.md
│   ├── dataset/                # Training data
│   │   ├── Training/
│   │   │   ├── tumor/
│   │   │   └── no_tumor/
│   │   └── Testing/
│   │       ├── tumor/
│   │       └── no_tumor/
│   ├── requirements.txt        # Python dependencies
│   ├── train.py                # Standalone training script
│   ├── TRAINING_GUIDE.md       # Training documentation
│   └── README.md
├── src/                         # React frontend source
│   ├── api/
│   │   └── client.ts           # API client
│   ├── assets/                 # Static assets
│   ├── components/             # React components
│   │   ├── ImageUpload.tsx
│   │   ├── PreprocessingSteps.tsx
│   │   ├── ProgressBar.tsx
│   │   ├── ResultPanel.tsx
│   │   └── Theme.ts
│   ├── styles/                 # CSS stylesheets
│   │   └── variables.css
│   ├── App.tsx                 # Main app component
│   └── main.tsx                # Application entry point
├── public/                      # Public assets
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── CODE_OF_CONDUCT.md          # Code of conduct
├── CONTRIBUTING.md             # Contributing guidelines
├── LICENSE                     # MIT License
├── README.md                   # This file
├── SECURITY.md                 # Security policy
├── eslint.config.js            # ESLint configuration
├── index.html                  # HTML entry point
├── package.json                # npm dependencies
├── package-lock.json           # npm lock file
├── tsconfig.json               # TypeScript configuration
├── tsconfig.app.json           # TypeScript app config
├── tsconfig.node.json          # TypeScript node config
└── vite.config.ts              # Vite configuration
```

## 📡 API Endpoints

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

## ⚙️ Configuration

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

### Color Palette

| Color | Hex | Usage |
|-------|-----|-------|
| Background | `#111` | Primary background |
| Text | `#e0e0e0` | Primary text |
| Info | `#4dd0e1` | Info accents, highlights |
| Danger | `#ff6b6b` | Tumor detection, errors |
| Success | `#00e676` | No tumor, success states |
| Warning | `#ffd93d` | Warnings |

## 📈 Performance

- **Training Time**: 30-60 minutes (GPU) for 10 epochs
- **Inference Time**: 1-2 seconds per image
- **Model Size**: ~350 MB
- **Accuracy**: Depends on dataset quality and size

## 🔧 Development

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

## 🐛 Troubleshooting

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

## 🤝 Contributing

We welcome contributions from the community! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- How to submit issues and pull requests
- Code style and standards
- Development workflow
- Testing requirements

Quick start:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🛡 Security

We take security seriously. If you discover a security vulnerability, please follow our [Security Policy](SECURITY.md) for responsible disclosure.

For security issues, please **DO NOT** open a public issue. Instead, report them via [GitHub Security Advisories](https://github.com/H0NEYP0T-466/MRI-BrainTumorClassification-DL/security).

## 📏 Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## 👥 Authors

- **H0NEYP0T-466** - *Creator and Maintainer*

## 🙏 Acknowledgments

- Vision Transformer (ViT) by Google Research
- timm library by Ross Wightman
- FastAPI framework
- React and Vite communities
- All contributors who help improve this project

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{mri_tumor_classification,
  title={MRI Brain Tumor Classification System},
  author={H0NEYP0T-466},
  year={2024},
  url={https://github.com/H0NEYP0T-466/MRI-BrainTumorClassification-DL}
}
```

## 💬 Support

- 📖 [Documentation](https://github.com/H0NEYP0T-466/MRI-BrainTumorClassification-DL#readme)
- 🐛 [Issue Tracker](https://github.com/H0NEYP0T-466/MRI-BrainTumorClassification-DL/issues)
- 💬 [Discussions](https://github.com/H0NEYP0T-466/MRI-BrainTumorClassification-DL/discussions)

---

<p align="center">
  <strong>⚠️ Important Notice</strong><br>
  This is a research/educational project. For clinical use, proper validation, regulatory approval, and medical expert supervision are required.
</p>

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/H0NEYP0T-466">H0NEYP0T-466</a>
</p>
