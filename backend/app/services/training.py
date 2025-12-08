"""Training pipeline with verbose logging."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm
import time

from app.logging_config import logger
from app.config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY
from app.services.model import create_model, save_model
from app.services.dataset import get_datasets


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        epoch: Current epoch number
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
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
        
        # Log every 10 batches
        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.4f}, "
                f"Accuracy: {100. * correct / total:.2f}%"
            )
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    epoch: int
) -> Dict[str, float]:
    """
    Validate model for one epoch.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        epoch: Current epoch number
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Validation {epoch}")
        
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
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def train_model(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE
) -> Dict[str, float]:
    """
    Train the model with verbose logging.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        Dictionary with final training metrics
    """
    logger.info("=" * 80)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info("=" * 80)
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset, test_dataset = get_datasets()
    
    if len(train_dataset) == 0:
        logger.error("Training dataset is empty!")
        raise ValueError("No training data available. Please ensure dataset is processed.")
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Testing samples: {len(test_dataset)}")
    
    # Create data loaders
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
    
    # Create model
    logger.info("Creating model...")
    model = create_model()
    
    # Loss function and optimizer
    logger.info("Setting up loss function and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    
    logger.info("\nStarting training loop...")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"EPOCH {epoch}/{epochs}")
        logger.info(f"{'=' * 80}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, epoch)
        logger.info(
            f"Epoch {epoch} Training - "
            f"Loss: {train_metrics['loss']:.4f}, "
            f"Accuracy: {train_metrics['accuracy']:.2f}%"
        )
        
        # Validate
        val_metrics = validate_epoch(model, test_loader, criterion, epoch)
        logger.info(
            f"Epoch {epoch} Validation - "
            f"Loss: {val_metrics['loss']:.4f}, "
            f"Accuracy: {val_metrics['accuracy']:.2f}%"
        )
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.6f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            save_model(model)
            logger.info(f"New best model saved! Validation accuracy: {best_val_acc:.2f}%")
        
        logger.info(f"{'=' * 80}\n")
    
    # Training complete
    elapsed_time = time.time() - start_time
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Total training time: {elapsed_time / 60:.2f} minutes")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    logger.info("=" * 80)
    
    return {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_train_acc': train_metrics['accuracy'],
        'final_val_acc': val_metrics['accuracy'],
        'training_time_minutes': elapsed_time / 60
    }
