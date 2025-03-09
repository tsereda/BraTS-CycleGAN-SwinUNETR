import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional
import time
from tqdm import tqdm

# Import the fixed loss and model
from losses import CombinedLoss
from swin_unetr import ImprovedSwinUNETR
from dataset import get_data_loaders, BraTSDataset


def train_model(
    data_path: str,
    output_path: str,
    batch_size: int = 1,
    num_workers: int = 2,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    use_mixed_precision: bool = True,
    gradient_accumulation_steps: int = 4,  # Increased for stability
    resume_from: Optional[str] = None
):
    """
    Main training function with improved stability
    
    Args:
        data_path: Path to dataset
        output_path: Path to save outputs
        batch_size: Batch size
        num_workers: Number of workers for data loading
        epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        use_mixed_precision: Whether to use mixed precision training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        resume_from: Path to checkpoint to resume from
    """
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        use_augmentation=True,
        debug=True  # Enable debug mode for the dataset
    )
    
    # Initialize model - Use simplified model architecture
    model = ImprovedSwinUNETR(
        in_channels=4,
        num_classes=4,
        feature_size=32  # Reduced feature size for stability
    )
    
    model.initialize_weights()
    model = model.to(device)
    
    # Print model parameters and structure
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Use fixed CombinedLoss with equal class weights
    class_weights = torch.tensor([0.25, 0.25, 0.25, 0.25]).to(device)
    loss_fn = CombinedLoss(
        dice_weight=0.5,
        focal_weight=0.5,
        class_weights=class_weights
    )
    
    # Optimizer with gradient clipping
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler - use simpler step scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,  # Reduce LR every 20 epochs
        gamma=0.5      # Reduce to 50% each time
    )
    
    # Gradient scaler for mixed precision
    scaler = torch.amp.GradScaler() if use_mixed_precision else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Checkpoint {resume_path} not found. Starting from scratch.")
    
    # Training loop
    print("\n=== Starting Training ===\n")
    
    for epoch in range(start_epoch, epochs):
        # Set to training mode
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        # Use tqdm for progress bar
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}') as pbar:
            # Reset gradients at the beginning of each epoch
            optimizer.zero_grad()
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                # Move data to device
                images, targets = images.to(device), targets.to(device)
                
                # Print data shapes and ranges for debugging
                if batch_idx == 0:
                    print(f"Image shape: {tuple(images.shape[2:])}, range: {images.min().item():.4f} to {images.max().item():.4f}")
                    print(f"Mask shape: {tuple(targets.shape)}, unique values: {torch.unique(targets).cpu().numpy()}")
                
                # Forward pass with mixed precision if enabled
                if use_mixed_precision:
                    with torch.amp.autocast("cuda"):
                        outputs = model(images)
                        loss = loss_fn(outputs, targets) / gradient_accumulation_steps
                    
                    # Backward pass with scaler
                    scaler.scale(loss).backward()
                    
                    # Update weights if we've accumulated enough gradients
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # Gradient clipping
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        print(f"Gradient norm: {grad_norm:.6f}")
                        
                        # Update weights
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # Standard forward pass
                    outputs = model(images)
                    loss = loss_fn(outputs, targets) / gradient_accumulation_steps
                    loss.backward()
                    
                    # Update weights if we've accumulated enough gradients
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        # Gradient clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        print(f"Gradient norm: {grad_norm:.6f}")
                        
                        # Update weights
                        optimizer.step()
                        optimizer.zero_grad()
                
                # Update running loss (use the scaled loss value)
                batch_loss = loss.item() * gradient_accumulation_steps  # Rescale for reporting
                train_loss += batch_loss
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                pbar.update()
        
        # Update scheduler
        scheduler.step()
        
        # Calculate average train loss
        train_loss /= batch_count
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
            }
            torch.save(checkpoint, output_path / f"model_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved to {output_path / f'model_epoch_{epoch+1}.pth'}")
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    print("\n=== Training Complete ===")
    return model


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Configuration
    config = {
        'data_path': "processed_data/brats128_split/",
        'output_path': "./output/",
        'batch_size': 1,
        'num_workers': 2,
        'epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'use_mixed_precision': True,
        'gradient_accumulation_steps': 4,
        'resume_from': None
    }
    
    # Train model
    model = train_model(**config)