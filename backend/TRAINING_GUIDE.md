# Training Guide

This guide explains how to train the MRI Brain Tumor Classification model using the standalone training script.

## Prerequisites

1. **Python 3.8+** installed
2. **Dependencies installed**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Dataset prepared** in the following structure:
   ```
   backend/dataset/
   ├── Training/
   │   ├── tumor/      # MRI scans with tumors
   │   └── no_tumor/   # MRI scans without tumors
   └── Testing/
       ├── tumor/
       └── no_tumor/
   ```

## Running the Training Script

### Basic Usage (Default Settings)

Simply run the training script with default parameters:

```bash
cd backend
python train.py
```

This will train the model with:
- **Epochs**: 10
- **Batch size**: 32
- **Learning rate**: 0.0001

### Custom Parameters

You can customize training parameters:

```bash
# Train for more epochs
python train.py --epochs 20

# Use smaller batch size (useful for limited memory)
python train.py --batch-size 16

# Adjust learning rate
python train.py --lr 0.00005

# Combine multiple parameters
python train.py --epochs 15 --batch-size 16 --lr 0.0001
```

### View Available Options

To see all available options:

```bash
python train.py --help
```

## What the Script Does

The training script performs the following steps with detailed logging:

### 1. Dataset Availability Check
- ✅ Verifies that raw dataset directories exist
- 📊 Counts images in each category
- ⚠️  Warns if any directories are missing

### 2. Dataset Preprocessing
- 📝 Checks if processed dataset exists
- 🔄 Preprocesses raw images if needed:
  - Denoising (Non-Local Means)
  - Contrast enhancement (CLAHE)
  - Edge sharpening
  - Brain segmentation
  - Intensity normalization
- ⏱️  Reports preprocessing time

### 3. Dataset Loading
- 📂 Creates PyTorch datasets
- 📊 Shows dataset statistics:
  - Number of training samples
  - Number of testing samples
  - Class distribution
- 🔄 Creates data loaders with specified batch size

### 4. Model Architecture
- 🧠 Displays model information:
  - Model name (Vision Transformer)
  - Input size
  - Number of classes
  - Device (CPU/GPU)
- 📈 Shows parameter counts and model size

### 5. Training Loop
For each epoch, the script:
- 🏋️ Trains on training data with progress bar
- ✅ Validates on testing data
- 📊 Reports metrics:
  - Training loss and accuracy
  - Validation loss and accuracy
  - Current learning rate
- 💾 Saves best model automatically

### 6. Training Summary
After training completes:
- ⏱️  Total training time
- 📈 Best validation accuracy achieved
- 📋 Complete training history table
- 💾 Model save location

## Output Example

```
================================================================================
         MRI BRAIN TUMOR CLASSIFICATION - MODEL TRAINING
================================================================================

⚙️  Training Configuration:
  Epochs:          10
  Batch size:      32
  Learning rate:   0.0001
  Weight decay:    1e-05
  Device:          cuda

================================================================================
  Dataset Availability Check
================================================================================
✓ Found      Training - Tumor              /path/to/dataset/Training/tumor
             └─ 1085 images
✓ Found      Training - No Tumor           /path/to/dataset/Training/no_tumor
             └─ 980 images
...

================================================================================
  Loading Datasets
================================================================================
📂 Creating PyTorch datasets...

✓ Datasets loaded successfully!
  Training samples:   2065
  Testing samples:     515
  Total samples:      2580

📊 Dataset Distribution:
  Training:
    No Tumor:   980 ( 47.5%)
    Tumor:     1085 ( 52.5%)
...

Epoch 1/10 [Train]: 100%|████████████| 65/65 [02:15<00:00, loss=0.3421, acc=84.23%]
Epoch 1/10 [Valid]: 100%|████████████| 17/17 [00:28<00:00, loss=0.2156, acc=91.46%]

📊 Epoch 1 Summary:
  Training   - Loss: 0.3421, Accuracy: 84.23%
  Validation - Loss: 0.2156, Accuracy: 91.46%
  Time: 163.5s
  Learning rate: 0.000100
  💾 New best model saved! Validation accuracy: 91.46%

...

================================================================================
                           TRAINING COMPLETED
================================================================================

📈 Final Results:
  Total training time:        45.23 minutes
  Average time per epoch:     271.4 seconds
  Best validation accuracy:   95.73% (Epoch 7)
  Final training accuracy:    97.12%
  Final validation accuracy:  95.53%
  Model saved to:             /path/to/backend/Model/model.pt

✨ Training completed successfully!
   You can now use the model for inference.
```

## Logging Details

The script provides comprehensive logging for every step:

### Dataset Loading
- ✅ Each image directory checked
- 📊 Image counts displayed
- ⚠️  Warnings for missing data

### Preprocessing
- 🔄 Progress bars for each processing step
- ⏱️  Time tracking
- ✅ Success confirmations

### Training Progress
- 📈 Real-time progress bars with loss/accuracy
- 📊 Epoch summaries
- 📉 Learning rate adjustments
- 💾 Best model save notifications

### Final Summary
- ⏱️  Complete timing information
- 📋 Full training history table
- 🎯 Best performance metrics

## Tips

1. **GPU Usage**: The script automatically uses GPU if available (CUDA). This significantly speeds up training.

2. **Memory Issues**: If you encounter out-of-memory errors:
   - Reduce batch size: `--batch-size 16` or `--batch-size 8`
   - Use CPU if GPU memory is insufficient

3. **Preprocessing Time**: The first run will take longer as it preprocesses all images. Subsequent runs will skip this step.

4. **Model Location**: The trained model is saved to `backend/Model/model.pt` and can be used for inference immediately.

5. **Interrupt Training**: Press `Ctrl+C` to stop training gracefully. The best model saved so far will be preserved.

## Troubleshooting

### Issue: `No module named 'torch'`
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: `Dataset directory does not exist`
**Solution**: Ensure your dataset is in the correct location (`backend/dataset/`) with the proper structure.

### Issue: `CUDA out of memory`
**Solution**: Reduce batch size or use CPU:
```bash
python train.py --batch-size 8
```

### Issue: Training is too slow
**Solution**: 
- Check if GPU is being used (look for `Device: cuda` in output)
- If using CPU, consider getting GPU access
- Reduce number of epochs for testing: `--epochs 5`

## Next Steps

After training completes:

1. **Check the model**: The trained model is at `backend/Model/model.pt`
2. **Use for inference**: The model can be used through the API or prediction interface
3. **Review metrics**: Check the training history table to understand model performance
4. **Fine-tune**: Adjust hyperparameters and retrain if needed

## Support

For issues or questions:
- Check the main README.md
- Review error messages carefully
- Ensure all prerequisites are met
