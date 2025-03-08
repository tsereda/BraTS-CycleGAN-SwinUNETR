import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable, Any
import random
import time
from tqdm import tqdm
import subprocess

# Import from custom modules
from swin_unetr_model import SwinUNETR
from dataset import get_data_loaders, BraTSDataset
from losses import CombinedLoss


def log_gpu_stats(step=None, batch_idx=None):
    """Log GPU statistics during training"""
    if not torch.cuda.is_available():
        return
    
    # Log at beginning of each epoch and periodically during an epoch
    if step is not None and batch_idx is not None and batch_idx % 10 != 0:
        return
    
    # Get GPU statistics
    gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
    gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
    gpu_max_memory = torch.cuda.max_memory_allocated() / (1024**3)    # GB
    
    # Get utilization using nvidia-smi (optional, might not work in all environments)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, 
            text=True
        )
        gpu_stats = [s.strip() for s in result.stdout.strip().split('\n')]
        gpu_info = []
        for i, stat in enumerate(gpu_stats):
            util, temp, mem_used, mem_total = map(float, stat.split(','))
            gpu_info.append(f"GPU {i}: {util:.1f}% util, {temp:.1f}°C, {mem_used/1024:.1f}/{mem_total/1024:.1f} GB")
    except:
        gpu_info = ["GPU stats unavailable"]
    
    # Log information
    step_info = f"Step: {step}" if step is not None else ""
    batch_info = f", Batch: {batch_idx}" if batch_idx is not None else ""
    print(f"\n--- GPU Stats ({step_info}{batch_info}) ---")
    print(f"PyTorch Memory: Allocated={gpu_memory_allocated:.2f}GB, Reserved={gpu_memory_reserved:.2f}GB, Peak={gpu_max_memory:.2f}GB")
    for info in gpu_info:
        print(info)


def compute_class_weights(data_path: str, use_equal_weights: bool = False) -> torch.Tensor:
    """
    Compute class weights based on class distribution
    
    Args:
        data_path: Path to dataset
        use_equal_weights: Whether to use equal weights
        
    Returns:
        Tensor of shape (4,) with class weights
    """
    if use_equal_weights:
        print("Using equal class weights: [0.25, 0.25, 0.25, 0.25]")
        return torch.tensor([0.25, 0.25, 0.25, 0.25])
    
    print("Calculating class weights...")
    train_path = Path(data_path) / 'train' / 'masks'
    val_path = Path(data_path) / 'val' / 'masks'
    
    mask_files = list(train_path.glob('*.npy')) + list(val_path.glob('*.npy'))
    
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {data_path}. Cannot compute class weights.")
    
    # Process a subset of files for speed
    if len(mask_files) > 50:
        mask_files = random.sample(mask_files, 50)
    
    class_counts = torch.zeros(4)
    
    for mask_file in mask_files:
        try:
            mask = np.load(mask_file, mmap_mode='r')
            mask_copy = mask.copy()
            
            # Convert one-hot to class indices if needed
            if len(mask_copy.shape) == 4 and mask_copy.shape[3] == 4:
                mask_copy = np.argmax(mask_copy, axis=3)
            
            # Count occurrences of each class
            for cls in range(4):
                class_counts[cls] += np.sum(mask_copy == cls)
                
        except Exception as e:
            print(f"Error loading mask file {mask_file}: {e}")
    
    # Ensure no zeros in class_counts
    class_counts = torch.clamp(class_counts, min=1.0)
    
    # Calculate weights
    total = class_counts.sum()
    weights = total / (4 * class_counts)
    
    # Clamp weights to avoid extreme values
    weights = torch.clamp(weights, min=0.1, max=50.0)
    
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {weights}")
    
    return weights


def validate(model, val_loader, loss_fn, device):
    """Validate the model on validation set"""
    model.eval()
    val_loss = 0.0
    ious = torch.zeros(4, device=device)  # IoU for each class
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
            
            # Calculate IoU for each class
            preds = torch.argmax(outputs, dim=1)
            for cls in range(4):
                intersection = torch.sum((preds == cls) & (targets == cls))
                union = torch.sum((preds == cls) | (targets == cls))
                iou = (intersection + 1e-6) / (union + 1e-6)
                ious[cls] += iou
    
    # Average over batches
    val_loss /= len(val_loader)
    ious /= len(val_loader)
    mean_iou = ious.mean().item()
    
    return val_loss, ious.cpu().numpy(), mean_iou


def save_checkpoint(model, optimizer, epoch, loss, mean_iou, filename):
    """Save model checkpoint"""
    # Handle distributed models
    model_to_save = model.module if hasattr(model, 'module') else model
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'mean_iou': mean_iou
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filename, map_location=device)
    
    # Handle distributed models
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    mean_iou = checkpoint.get('mean_iou', 0.0)  # Default to 0 if not present
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f} and IoU {mean_iou:.4f}")
    return model, optimizer, epoch, loss, mean_iou


def setup_distributed():
    """Set up distributed training"""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # Initialize the process group
        if 'WORLD_SIZE' in os.environ:
            # Using environment variables for multi-node setup
            dist.init_process_group(backend='nccl')
            rank = dist.get_rank()
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        else:
            # Single node, multi-GPU setup
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='tcp://localhost:12355',
                world_size=torch.cuda.device_count(),
                rank=0
            )
            rank = 0
            local_rank = 0
            
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        return True, device, rank, local_rank
    else:
        # Single GPU or CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return False, device, 0, 0


def train_model(
    data_path: str,
    output_path: str,
    batch_size: int = 2,
    num_workers: int = 0,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    use_equal_weights: bool = False,
    use_mixed_precision: bool = True,
    gradient_accumulation_steps: int = 2,
    resume_from: Optional[str] = None,
    use_distributed: bool = True,
):
    """
    Main training function with multi-GPU support
    
    Args:
        data_path: Path to dataset
        output_path: Path to save outputs
        batch_size: Batch size per GPU
        num_workers: Number of workers for data loading
        epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        use_equal_weights: Whether to use equal class weights
        use_mixed_precision: Whether to use mixed precision training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        resume_from: Path to checkpoint to resume from
        use_distributed: Whether to use distributed training
    """
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up distributed training
    is_distributed, device, rank, local_rank = setup_distributed() if use_distributed else (False, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 0, 0)
    
    # Print information about the environment
    print(f"Using device: {device}")
    if is_distributed:
        print(f"Distributed training enabled.")
        print(f"World size: {dist.get_world_size()}")
        print(f"Rank: {rank}, Local Rank: {local_rank}")
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(local_rank) / 1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(local_rank) / 1e9:.2f} GB")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Log initial GPU stats
    log_gpu_stats()
    
    # Get data loaders with distributed sampling if needed
    train_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset) if 'train_dataset' in locals() else None
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        use_augmentation=True,
        debug=False,
        train_sampler=train_sampler
    )
    
    # Test data loading time for first batch
    try:
        print("Testing data loading time for first batch...")
        start_time = time.time()
        first_batch = next(iter(train_loader))
        loading_time = time.time() - start_time
        print(f"First batch loaded in {loading_time:.2f} seconds")
        print(f"Batch shapes: images {first_batch[0].shape}, masks {first_batch[1].shape}")
        
        # Free memory
        del first_batch
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error loading first batch: {e}")
    
    # Calculate class weights
    class_weights = compute_class_weights(data_path, use_equal_weights)
    class_weights = class_weights.to(device)
    
    # Initialize model
    model = SwinUNETR(
        in_channels=4,
        num_classes=4,
        init_features=16  # This will be internally scaled
    )

    model.initialize_weights()
    model = model.to(device)
    
    # Wrap model in DDP if using distributed training
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"Model wrapped with DistributedDataParallel")
    
    # Loss function
    loss_fn = CombinedLoss(
        dice_weight=1.0,
        focal_weight=1.0,
        class_weights=class_weights
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double period after each restart
        eta_min=learning_rate / 100  # Min LR
    )
    
    # Gradient scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_mean_iou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'mean_iou': [], 'lr': []}
    
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            model, optimizer, start_epoch, _, best_mean_iou = load_checkpoint(
                model, optimizer, resume_path, device
            )
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Checkpoint {resume_path} not found. Starting from scratch.")
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        # Use tqdm for progress bar (only on rank 0 if distributed)
        if not is_distributed or (is_distributed and rank == 0):
            pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')
        else:
            pbar = None
        
        # Reset gradients at the beginning of each epoch
        optimizer.zero_grad()
        
        # Log GPU stats at beginning of epoch
        log_gpu_stats(epoch)
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass with mixed precision if enabled
            if use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = loss_fn(outputs, targets) / gradient_accumulation_steps
                
                # Backward pass with scaler
                scaler.scale(loss).backward()
                
                # Update weights if we've accumulated enough gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # Update weights
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Update running loss (use the scaled loss value)
            batch_loss = loss.item() * gradient_accumulation_steps  # Rescale for reporting
            train_loss += batch_loss
            batch_count += 1
            
            # Update progress bar (only on rank 0 if distributed)
            if pbar is not None:
                pbar.set_postfix({
                    'train_loss': f'{batch_loss:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                pbar.update()
            
            # Log GPU stats periodically
            if batch_idx % 50 == 0:
                log_gpu_stats(epoch, batch_idx)
        
        if pbar is not None:
            pbar.close()
        
        # Validate after each epoch (only on rank 0 if distributed)
        if not is_distributed or (is_distributed and rank == 0):
            val_loss, class_ious, mean_iou = validate(model, val_loader, loss_fn, device)
            
            # Update scheduler
            scheduler.step()
            
            # Calculate average train loss
            train_loss /= batch_count
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['mean_iou'].append(mean_iou)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Mean IoU: {mean_iou:.4f}")
            print(f"  Class IoUs: {class_ious}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if mean_iou > best_mean_iou:
                best_mean_iou = mean_iou
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    mean_iou=mean_iou,
                    filename=output_path / f"best_model.pth"
                )
            
            # Save latest model
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    mean_iou=mean_iou,
                    filename=output_path / f"model_epoch_{epoch+1}.pth"
                )
            
            # Plot training history
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                plot_training_history(history, output_path / "training_history.png")
        
        # Synchronize distributed processes
        if is_distributed:
            dist.barrier()
    
    return model, history


def plot_training_history(history, filename):
    """Plot training history"""
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot IoU and LR
    plt.subplot(2, 2, 3)
    plt.plot(history['mean_iou'], 'g-')
    plt.title('Mean IoU vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(history['lr'], 'r-')
    plt.title('Learning Rate vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Training history plot saved to {filename}")


def visualize_sample(model, data_path, output_path, sample_idx=0):
    """
    Visualize a sample prediction
    
    Args:
        model: Trained model
        data_path: Path to dataset
        output_path: Path to save visualization
        sample_idx: Index of sample to visualize
    """
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get a sample from validation set
    val_dataset = BraTSDataset(data_path, mode='val', transform=None)
    
    if not val_dataset.img_files:
        print("No validation samples found.")
        return
    
    # Get sample
    sample_idx = min(sample_idx, len(val_dataset) - 1)
    img, mask = val_dataset[sample_idx]
    
    # Add batch dimension and move to device
    img = img.unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    # Handle distributed models
    if hasattr(model, 'module'):
        model = model.module
    model = model.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Convert tensors to numpy arrays
    img = img.squeeze().cpu().numpy()
    mask = mask.cpu().numpy()
    
    # Visualize middle slice
    slice_idx = mask.shape[0] // 2
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(141)
    plt.title('FLAIR')
    plt.imshow(img[0, slice_idx], cmap='gray')
    plt.axis('off')
    
    plt.subplot(142)
    plt.title('T1CE')
    plt.imshow(img[1, slice_idx], cmap='gray')
    plt.axis('off')
    
    plt.subplot(143)
    plt.title('Ground Truth')
    plt.imshow(mask[slice_idx])
    plt.axis('off')
    
    plt.subplot(144)
    plt.title('Prediction')
    plt.imshow(pred[slice_idx])
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / f"prediction_sample_{sample_idx}.png")
    plt.close()
    
    print(f"Visualization saved to {output_path / f'prediction_sample_{sample_idx}.png'}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Configuration
    config = {
        'data_path': "processed_data/brats128_split/",
        'output_path': "/tmp/output/",
        'batch_size': 2,             # Increased from 1 to 2 per GPU
        'num_workers': 2,            # Careful with this value - it's per GPU
        'epochs': 100,
        'learning_rate': 5e-4,
        'weight_decay': 1e-5,
        'use_equal_weights': False,
        'use_mixed_precision': True,
        'gradient_accumulation_steps': 2,
        'resume_from': None,
        'use_distributed': True      # Enable distributed training
    }
    
    # Train model
    model, history = train_model(**config)
    
    # Visualize samples (only if rank 0 in distributed setting)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        for i in range(3):
            visualize_sample(
                model=model,
                data_path=config['data_path'],
                output_path=config['output_path'],
                sample_idx=i
            )