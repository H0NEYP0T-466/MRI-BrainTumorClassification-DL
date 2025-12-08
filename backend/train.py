#!/usr/bin/env python3
"""
Standalone training script for MRI Brain Tumor Classification.

This script can be run directly to train the model without needing to start the API server.
It includes comprehensive logging at every step:
- Dataset loading
- Dataset preprocessing
- Model architecture details
- Training progress (epochs, batches, loss, accuracy)
- Validation metrics
- Final results

Usage:
    python train.py
    python train.py --epochs 20 --batch-size 16
"""

import argparse
import sys
import time
from pathlib import Path

# Add the backend directory to Python path for imports
BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.config import (
    DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    TRAIN_RAW_TUMOR, TRAIN_RAW_NO_TUMOR,
    TEST_RAW_TUMOR, TEST_RAW_NO_TUMOR,
    TRAIN_PROCESSED_TUMOR, TRAIN_PROCESSED_NO_TUMOR,
    TEST_PROCESSED_TUMOR, TEST_PROCESSED_NO_TUMOR,
    MODEL_PATH, IMAGE_SIZE, NUM_CLASSES, MODEL_NAME
)
from app.services.model import BrainTumorClassifier, save_model
from app.services.dataset import get_datasets, process_dataset


def print_banner(text: str, char: str = "="):
    """Print a formatted banner."""
    width = 80
    print("\n" + char * width)
    print(text.center(width))
    print(char * width)


def print_section(text: str):
    """Print a section header."""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}")


def check_dataset_exists():
    """Check if raw dataset exists and print status."""
    print_section("Dataset Availability Check")
    
    dataset_paths = [
        (TRAIN_RAW_TUMOR, "Training - Tumor"),
        (TRAIN_RAW_NO_TUMOR, "Training - No Tumor"),
        (TEST_RAW_TUMOR, "Testing - Tumor"),
        (TEST_RAW_NO_TUMOR, "Testing - No Tumor"),
    ]
    
    all_exist = True
    for path, label in dataset_paths:
        exists = path.exists()
        status = "✓ Found" if exists else "✗ Missing"
        print(f"{status:12} {label:30} {path}")
        if exists:
            # Count images
            image_files = list(path.glob("*.png")) + list(path.glob("*.jpg")) + list(path.glob("*.jpeg"))
            print(f"             └─ {len(image_files)} images")
        all_exist = all_exist and exists
    
    if not all_exist:
        print("\n⚠️  WARNING: Some dataset directories are missing!")
        print("   Please ensure your dataset is properly organized.")
    
    return all_exist


def preprocess_dataset():
    """Preprocess the dataset if not already processed."""
    print_section("Dataset Preprocessing")
    
    # Check if processed dataset exists
    processed_paths = [
        TRAIN_PROCESSED_TUMOR,
        TRAIN_PROCESSED_NO_TUMOR,
        TEST_PROCESSED_TUMOR,
        TEST_PROCESSED_NO_TUMOR,
    ]
    
    needs_processing = False
    for path in processed_paths:
        if not path.exists() or len(list(path.glob("*.png"))) == 0:
            needs_processing = True
            break
    
    if needs_processing:
        print("📝 Processed dataset not found or incomplete.")
        print("🔄 Starting preprocessing pipeline...")
        print("   This includes:")
        print("   - Denoising (Non-Local Means)")
        print("   - Contrast enhancement (CLAHE)")
        print("   - Edge sharpening")
        print("   - Brain segmentation")
        print("   - Intensity normalization")
        print()
        
        start_time = time.time()
        process_dataset(force_reprocess=False)
        elapsed = time.time() - start_time
        
        print(f"\n✓ Preprocessing completed in {elapsed / 60:.2f} minutes")
    else:
        print("✓ Processed dataset already exists. Skipping preprocessing.")
        
        # Show processed dataset statistics
        print("\nProcessed dataset statistics:")
        for path in processed_paths:
            image_count = len(list(path.glob("*.png")))
            print(f"  {path.name:15} {image_count:5} images")


def load_datasets(batch_size: int):
    """Load and prepare datasets."""
    print_section("Loading Datasets")
    
    print("📂 Creating PyTorch datasets...")
    train_dataset, test_dataset = get_datasets()
    
    print(f"\n✓ Datasets loaded successfully!")
    print(f"  Training samples:   {len(train_dataset):5}")
    print(f"  Testing samples:    {len(test_dataset):5}")
    print(f"  Total samples:      {len(train_dataset) + len(test_dataset):5}")
    
    if len(train_dataset) == 0:
        print("\n❌ ERROR: Training dataset is empty!")
        print("   Please ensure your dataset is properly set up.")
        sys.exit(1)
    
    if len(test_dataset) == 0:
        print("\n❌ ERROR: Testing dataset is empty!")
        print("   Please ensure your dataset is properly set up.")
        sys.exit(1)
    
    # Calculate class distribution efficiently in one pass
    print("\n📊 Dataset Distribution:")
    train_no_tumor = 0
    train_tumor = 0
    for _, label in train_dataset.data:
        if label == 0:
            train_no_tumor += 1
        else:
            train_tumor += 1
    
    test_no_tumor = 0
    test_tumor = 0
    for _, label in test_dataset.data:
        if label == 0:
            test_no_tumor += 1
        else:
            test_tumor += 1
    
    print(f"  Training:")
    print(f"    No Tumor: {train_no_tumor:5} ({100 * train_no_tumor / len(train_dataset):5.1f}%)")
    print(f"    Tumor:    {train_tumor:5} ({100 * train_tumor / len(train_dataset):5.1f}%)")
    print(f"  Testing:")
    print(f"    No Tumor: {test_no_tumor:5} ({100 * test_no_tumor / len(test_dataset):5.1f}%)")
    print(f"    Tumor:    {test_tumor:5} ({100 * test_tumor / len(test_dataset):5.1f}%)")
    
    # Create data loaders
    print(f"\n🔄 Creating data loaders with batch size: {batch_size}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"  Training batches:   {len(train_loader):5}")
    print(f"  Testing batches:    {len(test_loader):5}")
    
    return train_loader, test_loader


def create_and_display_model():
    """Create model and display architecture details."""
    print_section("Model Architecture")
    
    print(f"🧠 Model: {MODEL_NAME}")
    print(f"📐 Input size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"🎯 Number of classes: {NUM_CLASSES}")
    print(f"💻 Device: {DEVICE}")
    
    print("\n🏗️  Creating model...")
    model = BrainTumorClassifier()
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Model created successfully!")
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    print(f"  Model size:            ~{total_params * 4 / (1024 ** 2):.1f} MB")
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{total_epochs} [Train]",
        leave=True,
        ncols=100
    )
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate_epoch(model, test_loader, criterion, epoch, total_epochs):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(
            test_loader,
            desc=f"Epoch {epoch}/{total_epochs} [Valid]",
            leave=True,
            ncols=100
        )
        
        for images, labels in progress_bar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_model(epochs: int, batch_size: int, learning_rate: float):
    """Main training function."""
    print_banner("MRI BRAIN TUMOR CLASSIFICATION - MODEL TRAINING")
    
    print(f"\n⚙️  Training Configuration:")
    print(f"  Epochs:          {epochs}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  Weight decay:    {WEIGHT_DECAY}")
    print(f"  Device:          {DEVICE}")
    
    # Step 1: Check dataset
    dataset_exists = check_dataset_exists()
    if not dataset_exists:
        print("\n❌ ERROR: Dataset not found. Exiting.")
        return
    
    # Step 2: Preprocess dataset
    preprocess_dataset()
    
    # Step 3: Load datasets
    train_loader, test_loader = load_datasets(batch_size)
    
    # Step 4: Create model
    model = create_and_display_model()
    
    # Step 5: Setup training
    print_section("Training Setup")
    
    print("📋 Loss function: CrossEntropyLoss")
    criterion = nn.CrossEntropyLoss()
    
    print(f"🎯 Optimizer: AdamW (lr={learning_rate}, weight_decay={WEIGHT_DECAY})")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )
    
    print("📉 Learning rate scheduler: ReduceLROnPlateau")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # Step 6: Training loop
    print_banner("TRAINING STARTED", "-")
    
    best_val_acc = 0.0
    best_epoch = 0
    training_history = []
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        print(f"\n{'=' * 80}")
        print(f"  EPOCH {epoch}/{epochs}")
        print(f"{'=' * 80}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, epoch, epochs
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, test_loader, criterion, epoch, epochs
        )
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"  Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"  📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        else:
            print(f"  Learning rate: {new_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_model(model)
            print(f"  💾 New best model saved! Validation accuracy: {best_val_acc:.2f}%")
        else:
            print(f"  Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
        
        # Store history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': new_lr
        })
    
    # Training complete
    total_time = time.time() - start_time
    
    print_banner("TRAINING COMPLETED", "=")
    
    print(f"\n📈 Final Results:")
    print(f"  Total training time:        {total_time / 60:.2f} minutes")
    print(f"  Average time per epoch:     {total_time / epochs:.1f} seconds")
    print(f"  Best validation accuracy:   {best_val_acc:.2f}% (Epoch {best_epoch})")
    if training_history:
        print(f"  Final training accuracy:    {training_history[-1]['train_acc']:.2f}%")
        print(f"  Final validation accuracy:  {training_history[-1]['val_acc']:.2f}%")
    print(f"  Model saved to:             {MODEL_PATH}")
    
    # Print training history table
    if training_history:
        print(f"\n{'=' * 80}")
        print("  TRAINING HISTORY")
        print(f"{'=' * 80}")
        print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>11} {'Val Loss':>10} {'Val Acc':>9} {'LR':>12}")
        print("-" * 80)
        for entry in training_history:
            print(
                f"{entry['epoch']:>6} "
                f"{entry['train_loss']:>12.4f} "
                f"{entry['train_acc']:>10.2f}% "
                f"{entry['val_loss']:>10.4f} "
                f"{entry['val_acc']:>8.2f}% "
                f"{entry['lr']:>12.6f}"
            )
        print("=" * 80)
    
    print(f"\n✨ Training completed successfully!")
    print(f"   You can now use the model for inference.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train MRI Brain Tumor Classification model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                           # Use default settings
  python train.py --epochs 20               # Train for 20 epochs
  python train.py --epochs 15 --batch-size 16 --lr 0.0001
        """
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help=f'Number of training epochs (default: {EPOCHS})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size for training (default: {BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=LEARNING_RATE,
        dest='learning_rate',
        help=f'Learning rate (default: {LEARNING_RATE})'
    )
    
    args = parser.parse_args()
    
    try:
        train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
