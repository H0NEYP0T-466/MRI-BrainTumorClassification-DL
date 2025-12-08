# Model Directory

This directory contains the trained model weights.

## Model File

- **Filename**: `model.pt`
- **Format**: PyTorch state dict
- **Architecture**: Vision Transformer (vit_base_patch16_224)
- **Size**: ~350 MB

## Contents

The model file contains:
- `model_state_dict`: Trained model weights
- `model_name`: Architecture identifier
- `num_classes`: Number of output classes (2)

## Usage

The model is automatically loaded on server startup if it exists.

To manually load:
```python
from app.services.model import load_model
model = load_model()
```

## Training

To train a new model, use the training endpoint:
```bash
POST /train
```

The best model will be automatically saved here during training.
