import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import time
import datetime
from tqdm import tqdm
import sys
import gc
import argparse
import numpy as np

# Import the fixed loss and model
from losses import CombinedLoss
from swin_unetr import ImprovedSwinUNETR
from dataset import get_data_loaders, BraTSDataset
from torch.optim.lr_scheduler import OneCycleLR


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BraTS Segmentation Training')
    parser.add_argument('--data_path', type=str,
                        help='Path to dataset')
    parser.add_argument('--output_path', type=str, default="output/",
                        help='Path to save outputs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Number of steps to accumulate gradients')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                      num_classes: int = 4) -> Tuple[List[float], List[float]]:
    """
    Calculate per-class Dice and MIoU metrics
    
    Args:
        predictions: Tensor of shape [B, C, D, H, W] with class logits
        targets: Tensor of shape [B, D, H, W] with class indices
        num_classes: Number of classes including background
        
    Returns:
        tuple containing:
            - dice_scores: List of Dice scores for each class
            - miou_scores: List of MIoU scores for each class
    """
    # Move to CPU and convert to numpy for metrics calculation
    if predictions.is_cuda:
        predictions = predictions.detach().cpu()
    if targets.is_cuda:
        targets = targets.detach().cpu()
    
    # Get predictions by taking the argmax
    predictions = torch.argmax(predictions, dim=1)
    
    batch_size = predictions.shape[0]
    dice_scores = [0.0] * num_classes
    iou_scores = [0.0] * num_classes
    
    # Calculate metrics for each class
    for class_idx in range(num_classes):
        # Create binary masks for the current class
        pred_mask = (predictions == class_idx).float()
        target_mask = (targets == class_idx).float()
        
        # Calculate intersection and union
        intersection = torch.sum(pred_mask * target_mask, dim=(1, 2, 3))
        pred_sum = torch.sum(pred_mask, dim=(1, 2, 3))
        target_sum = torch.sum(target_mask, dim=(1, 2, 3))
        union = pred_sum + target_sum - intersection
        
        # Calculate Dice score: 2*intersection / (pred_sum + target_sum)
        # Add small constant to avoid division by zero
        dice_batch = (2.0 * intersection + 1e-5) / (pred_sum + target_sum + 1e-5)
        
        # Calculate IoU: intersection / union
        iou_batch = (intersection + 1e-5) / (union + 1e-5)
        
        # Average over the batch
        dice_scores[class_idx] = dice_batch.mean().item()
        iou_scores[class_idx] = iou_batch.mean().item()
    
    return dice_scores, iou_scores


def validate(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, 
             device: torch.device, num_classes: int = 4) -> Dict:
    """
    Validate the model on the validation set
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        device: Device to run validation on
        num_classes: Number of classes including background
        
    Returns:
        Dict with validation metrics
    """
    model.eval()
    val_dice_scores = [[] for _ in range(num_classes)]
    val_iou_scores = [[] for _ in range(num_classes)]
    
    print("Starting validation...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            # Move data to device
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate metrics
            dice_scores, iou_scores = calculate_metrics(outputs, targets, num_classes)
            
            # Append batch metrics to lists
            for class_idx in range(num_classes):
                val_dice_scores[class_idx].append(dice_scores[class_idx])
                val_iou_scores[class_idx].append(iou_scores[class_idx])
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                print(f"Validated {batch_idx + 1}/{len(val_loader)} batches")
                
    # Calculate mean metrics across all batches
    mean_dice = [sum(scores) / max(len(scores), 1) for scores in val_dice_scores]
    mean_iou = [sum(scores) / max(len(scores), 1) for scores in val_iou_scores]
    
    # Calculate mean across all classes
    overall_dice = sum(mean_dice) / len(mean_dice)
    overall_miou = sum(mean_iou) / len(mean_iou)
    
    # Prepare metrics dictionary
    metrics = {
        'dice_per_class': mean_dice,
        'miou_per_class': mean_iou,
        'overall_dice': overall_dice,
        'overall_miou': overall_miou
    }
    
    return metrics


def train_model(
    data_path: str,
    output_path: str,
    batch_size: int = 1,  # Changed from 2 to 1
    num_workers: int = 2,  # Changed from 4 to 2
    epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    use_mixed_precision: bool = True,
    gradient_accumulation_steps: int = 2,  # Changed from 1 to 2 to compensate for smaller batch size
    resume_from: Optional[str] = None,
    benchmark_mode: bool = True  # Enable cuDNN benchmarking
):
    """
    Optimized training function for A100 GPUs with memory efficiency improvements
    """
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set device and optimization flags
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print shared memory information
    if os.path.exists('/dev/shm'):
        shm_stats = os.statvfs('/dev/shm')
        shm_size_gb = shm_stats.f_blocks * shm_stats.f_frsize / (1024**3)
        shm_free_gb = shm_stats.f_bavail * shm_stats.f_frsize / (1024**3)
        print(f"Shared memory (/dev/shm): Total: {shm_size_gb:.2f} GB, Free: {shm_free_gb:.2f} GB")
    
    # Enable cuDNN benchmarking and deterministic algorithms for A100
    if benchmark_mode and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("cuDNN benchmark mode enabled for optimized performance")
    
    print(f"Using device: {device}")
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
        print(f"PyTorch CUDA current device: {torch.cuda.current_device()}")
        print(f"Initial Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Initial Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        # Reset max memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Check if Tensor Cores can be utilized (A100 has compute capability 8.0)
        if hasattr(torch.cuda, 'get_device_capability'):
            capability = torch.cuda.get_device_capability(0)
            print(f"CUDA Compute Capability: {capability[0]}.{capability[1]}")
            if capability[0] >= 8:
                print("GPU supports Tensor Cores (A100 or newer)")
                print("Optimizing for Tensor Core operations...")
                
                # Set tensor core optimization flags for PyTorch
                torch.set_float32_matmul_precision('high')
                print("Tensor Core optimization enabled (float32_matmul_precision=high)")
    
    # Get data loaders with optimized settings
    train_loader, val_loader = get_data_loaders(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,  # Use fewer workers to reduce shared memory usage
        use_augmentation=True,
        debug=True  # Enable debug mode for the dataset
    )
    
    # Initialize model with slightly reduced feature size to save memory
    model = ImprovedSwinUNETR(
        in_channels=4,
        num_classes=4,
        feature_size=48  # Reduced from 48 to 32 to save memory
    )
    
    model.initialize_weights()
    model = model.to(device)
    
    # Print model parameters and structure
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Use fixed CombinedLoss with equal class weights
    class_weights = torch.tensor([0.1, 0.45, 0.2, 0.25]).to(device)
    loss_fn = CombinedLoss(
        dice_weight=0.5,
        focal_weight=0.5,
        class_weights=class_weights
    )
    
    # Optimizer with gradient clipping
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False
    )
    
    # Learning rate scheduler - use cosine annealing for better convergence
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate * 5,  # Peak LR 5x initial LR
        steps_per_epoch=len(train_loader) // gradient_accumulation_steps,
        epochs=epochs,
        pct_start=0.3,  # Spend 30% of time warming up
        div_factor=25,   # Initial LR is max_lr/25
        final_div_factor=10000  # Final LR is max_lr/10000
    )
    
    # Gradient scaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=use_mixed_precision)
    
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
    
    # Create metrics log file
    metrics_log_path = output_path / 'metrics.csv'
    with open(metrics_log_path, 'w') as f:
        f.write("epoch,train_loss,overall_dice,overall_miou,background_dice,et_dice,tc_dice,wt_dice,background_iou,et_iou,tc_iou,wt_iou\n")
    
    # Training loop
    print("\n=== Starting Training ===\n")
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        # Set to training mode
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        # Print epoch information
        print(f"Epoch {epoch+1}/{epochs} started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Reset gradients at the beginning of each epoch
        optimizer.zero_grad(set_to_none=True)
        
        try:
            for batch_idx, (images, targets) in enumerate(train_loader):
                batch_start_time = time.time()
                
                # Move data to device
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                # Clear unnecessary CPU tensors
                torch.cuda.empty_cache()
                
                # Forward pass with mixed precision if enabled
                with torch.amp.autocast(device_type="cuda", enabled=use_mixed_precision):
                    outputs = model(images)
                    loss = loss_fn(outputs, targets) / gradient_accumulation_steps
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
                
                # Update weights if we've accumulated enough gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update weights
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                
                # Update running loss (use the scaled loss value)
                batch_loss = loss.item() * gradient_accumulation_steps
                train_loss += batch_loss
                batch_count += 1
                
                # Calculate batch time
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                
                # Clear unnecessary GPU tensors
                del outputs
                torch.cuda.empty_cache()
                
                # Print progress
                if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                    current_memory = torch.cuda.memory_allocated(0) / 1e9
                    max_memory = torch.cuda.max_memory_allocated(0) / 1e9
                    
                    print(f"Loss: {batch_loss:.4f}, "
                          f"Time: {batch_time:.3f}s, "
                          f"GPU Mem: {current_memory:.2f}/{max_memory:.2f} GB, "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                    
                # Ensure output is flushed immediately
                sys.stdout.flush()
                
        except Exception as e:
            print(f"Error during training: {e}")
            # Save emergency checkpoint
            emergency_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'error': str(e)
            }
            torch.save(emergency_checkpoint, output_path / f"emergency_checkpoint_epoch_{epoch+1}.pth")
            print(f"Emergency checkpoint saved to {output_path / f'emergency_checkpoint_epoch_{epoch+1}.pth'}")
            raise e
        
        # Update scheduler
        scheduler.step()
        
        # Calculate average train loss
        if batch_count > 0:
            train_loss /= batch_count
        
        # Run validation after the first epoch and every 10 epochs
        validation_metrics = {}
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"\nRunning validation after epoch {epoch+1}")
            validation_metrics = validate(model, val_loader, device)
            
            # Print validation metrics
            print("\nValidation Metrics:")
            print(f"  Overall Dice Score: {validation_metrics['overall_dice']:.4f}")
            print(f"  Overall MIoU: {validation_metrics['overall_miou']:.4f}")
            print("\nDice Scores per Class:")
            class_names = ["Background", "Enhancing Tumor", "Tumor Core", "Whole Tumor"]
            for i, cls_name in enumerate(class_names):
                print(f"  {cls_name}: {validation_metrics['dice_per_class'][i]:.4f}")
            
            print("\nMIoU Scores per Class:")
            for i, cls_name in enumerate(class_names):
                print(f"  {cls_name}: {validation_metrics['miou_per_class'][i]:.4f}")
        
        # Calculate epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss
            }
            
            # Add validation metrics if calculated
            if validation_metrics:
                checkpoint['dice_score'] = validation_metrics['overall_dice']
                checkpoint['miou'] = validation_metrics['overall_miou']
            
            torch.save(checkpoint, output_path / f"model_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved to {output_path / f'model_epoch_{epoch+1}.pth'}")
        
        # Log metrics to CSV
        with open(metrics_log_path, 'a') as f:
            # If validation was run
            if validation_metrics:
                dice_per_class = validation_metrics['dice_per_class']
                miou_per_class = validation_metrics['miou_per_class']
                f.write(f"{epoch+1},{train_loss:.6f},{validation_metrics['overall_dice']:.6f},"
                        f"{validation_metrics['overall_miou']:.6f},{dice_per_class[0]:.6f},"
                        f"{dice_per_class[1]:.6f},{dice_per_class[2]:.6f},{dice_per_class[3]:.6f},"
                        f"{miou_per_class[0]:.6f},{miou_per_class[1]:.6f},{miou_per_class[2]:.6f},"
                        f"{miou_per_class[3]:.6f}\n")
            else:
                # If validation was not run this epoch, just log training loss
                f.write(f"{epoch+1},{train_loss:.6f},,,,,,,,,\n")
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Epoch Time: {epoch_time:.2f}s ({time.strftime('%H:%M:%S', time.gmtime(epoch_time))})")
        print(f"  Current GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Max GPU Memory: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        
        # Force clear CUDA cache after each epoch to avoid memory fragmentation
        torch.cuda.empty_cache()
        gc.collect()
        
        sys.stdout.flush()
    
    print("\n=== Training Complete ===")
    
    return model


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Parse command line arguments
    args = parse_args()
    
    # Configuration for A100 GPU
    config = {
        'data_path': args.data_path,
        'output_path': args.output_path,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'use_mixed_precision': True,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'resume_from': args.resume_from,
        'benchmark_mode': True
    }
    
    # Train model
    model = train_model(**config)