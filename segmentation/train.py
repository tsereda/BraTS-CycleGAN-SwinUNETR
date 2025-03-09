import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional
import time
import datetime
from tqdm import tqdm
import sys
import gc

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
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
        print(f"PyTorch CUDA current device: {torch.cuda.current_device()}")
        print(f"Initial Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Initial Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        
        # Reset max memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Check compute capability for A100 (compute capability 8.0)
        if hasattr(torch.cuda, 'get_device_capability'):
            capability = torch.cuda.get_device_capability(0)
            print(f"CUDA Compute Capability: {capability[0]}.{capability[1]}")
            if capability[0] >= 8:
                print("GPU supports Tensor Cores (A100 or newer)")
            
        # Run a quick benchmark
        print("\n--- Running quick GPU benchmark ---")
        benchmark_tensor_size = 5000
        start_time = time.time()
        x = torch.randn(benchmark_tensor_size, benchmark_tensor_size, device=device)
        y = torch.randn(benchmark_tensor_size, benchmark_tensor_size, device=device)
        torch.cuda.synchronize()
        matmul_start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        matmul_end = time.time()
        print(f"Matrix multiplication time ({benchmark_tensor_size}x{benchmark_tensor_size}): {(matmul_end - matmul_start):.4f} seconds")
        del x, y, z
        torch.cuda.empty_cache()
        gc.collect()
    
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
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
    
    # Track timing statistics
    epoch_times = []
    batch_times = []
    forward_times = []
    backward_times = []
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        # Set to training mode
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        # Force tqdm to use a basic progress bar that works in all environments
        print(f"Epoch {epoch+1}/{epochs} started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Reset gradients at the beginning of each epoch
        optimizer.zero_grad()
        
        # Timing stats for this epoch
        epoch_batch_times = []
        epoch_forward_times = []
        epoch_backward_times = []
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            batch_start_time = time.time()
            
            # Move data to device
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass with mixed precision if enabled
            forward_start_time = time.time()
            if use_mixed_precision:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = loss_fn(outputs, targets) / gradient_accumulation_steps
                
                forward_end_time = time.time()
                forward_time = forward_end_time - forward_start_time
                epoch_forward_times.append(forward_time)
                
                # Backward pass with scaler
                backward_start_time = time.time()
                scaler.scale(loss).backward()
                
                # Update weights if we've accumulated enough gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update weights
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                backward_end_time = time.time()
                backward_time = backward_end_time - backward_start_time
                epoch_backward_times.append(backward_time)
                
            else:
                # Standard forward pass
                outputs = model(images)
                loss = loss_fn(outputs, targets) / gradient_accumulation_steps
                
                forward_end_time = time.time()
                forward_time = forward_end_time - forward_start_time
                epoch_forward_times.append(forward_time)
                
                # Backward pass
                backward_start_time = time.time()
                loss.backward()
                
                # Update weights if we've accumulated enough gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update weights
                    optimizer.step()
                    optimizer.zero_grad()
                
                backward_end_time = time.time()
                backward_time = backward_end_time - backward_start_time
                epoch_backward_times.append(backward_time)
            
            # Update running loss (use the scaled loss value)
            batch_loss = loss.item() * gradient_accumulation_steps  # Rescale for reporting
            train_loss += batch_loss
            batch_count += 1
            
            # Calculate batch time
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            epoch_batch_times.append(batch_time)
            
            # Print progress explicitly instead of using tqdm
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:  # Print more frequently
                # Get current GPU memory usage
                current_memory = torch.cuda.memory_allocated(0) / 1e9
                max_memory = torch.cuda.max_memory_allocated(0) / 1e9
                
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {batch_loss:.4f}, "
                      f"Time: {batch_time:.3f}s (F: {forward_time:.3f}s, B: {backward_time:.3f}s), "
                      f"GPU Mem: {current_memory:.2f}/{max_memory:.2f} GB, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                
            # Ensure output is flushed immediately in containerized environments
            sys.stdout.flush()
        
        # Update scheduler
        scheduler.step()
        
        # Calculate average train loss
        train_loss /= batch_count
        
        # Calculate epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Calculate average batch, forward, and backward times
        avg_batch_time = sum(epoch_batch_times) / len(epoch_batch_times)
        avg_forward_time = sum(epoch_forward_times) / len(epoch_forward_times)
        avg_backward_time = sum(epoch_backward_times) / len(epoch_backward_times)
        
        batch_times.append(avg_batch_time)
        forward_times.append(avg_forward_time)
        backward_times.append(avg_backward_time)
        
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
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Epoch Time: {epoch_time:.2f}s ({time.strftime('%H:%M:%S', time.gmtime(epoch_time))})")
        print(f"  Avg Batch Time: {avg_batch_time:.3f}s")
        print(f"  Avg Forward Time: {avg_forward_time:.3f}s")
        print(f"  Avg Backward Time: {avg_backward_time:.3f}s")
        print(f"  Current GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Max GPU Memory: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        
        # Force clear CUDA cache periodically to avoid memory fragmentation
        if (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
        
        sys.stdout.flush()
    
    # Print overall timing statistics
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_batch_time_overall = sum(batch_times) / len(batch_times)
    avg_forward_time_overall = sum(forward_times) / len(forward_times)
    avg_backward_time_overall = sum(backward_times) / len(backward_times)
    
    print("\n=== Training Complete ===")
    print(f"Average Epoch Time: {avg_epoch_time:.2f}s ({time.strftime('%H:%M:%S', time.gmtime(avg_epoch_time))})")
    print(f"Average Batch Time: {avg_batch_time_overall:.3f}s")
    print(f"Average Forward Time: {avg_forward_time_overall:.3f}s")
    print(f"Average Backward Time: {avg_backward_time_overall:.3f}s")
    print(f"Final GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
    
    return model


def diagnose_gpu_performance():
    """Run a series of tests to diagnose GPU performance issues"""
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot diagnose GPU performance.")
        return
    
    print("\n=== GPU Performance Diagnostics ===")
    
    # Basic info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check compute capability
    if hasattr(torch.cuda, 'get_device_capability'):
        capability = torch.cuda.get_device_capability(0)
        print(f"CUDA Compute Capability: {capability[0]}.{capability[1]}")
    
    # Test matrix multiplication (CPU vs GPU)
    sizes = [1000, 2000, 4000]
    
    for size in sizes:
        print(f"\nTesting matrix multiplication ({size}x{size}):")
        
        # CPU test
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        cpu_start = time.time()
        _ = torch.matmul(a_cpu, b_cpu)
        cpu_end = time.time()
        cpu_time = cpu_end - cpu_start
        
        # GPU test
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        
        # Warmup
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        gpu_start = time.time()
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_end = time.time()
        gpu_time = gpu_end - gpu_start
        
        speedup = cpu_time / gpu_time
        
        print(f"  CPU Time: {cpu_time:.4f}s")
        print(f"  GPU Time: {gpu_time:.4f}s")
        print(f"  Speedup: {speedup:.1f}x")
        
        # Clean up
        del a_cpu, b_cpu, a_gpu, b_gpu, _
        torch.cuda.empty_cache()
    
    # Test memory bandwidth
    print("\nTesting memory bandwidth:")
    tensor_size = 10000000  # 10M elements
    
    # Create large tensor
    x_gpu = torch.randn(tensor_size, device='cuda')
    y_gpu = torch.randn(tensor_size, device='cuda')
    
    # Warmup
    _ = x_gpu + y_gpu
    torch.cuda.synchronize()
    
    # Measure bandwidth
    iterations = 100
    start_time = time.time()
    for _ in range(iterations):
        _ = x_gpu + y_gpu
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate bandwidth (bytes read + written per second)
    # Each operation reads 2 tensors and writes 1 tensor
    elapsed = end_time - start_time
    elements_processed = tensor_size * iterations
    bytes_processed = elements_processed * 4 * 3  # 4 bytes per float32, 3 operations (2 read, 1 write)
    bandwidth = bytes_processed / elapsed / (1024 ** 3)  # Convert to GB/s
    
    print(f"  Memory Bandwidth: {bandwidth:.2f} GB/s")
    
    # Clean up
    del x_gpu, y_gpu, _
    torch.cuda.empty_cache()
    
    print("\nGPU Performance Diagnostics Complete")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Run diagnostics first
    diagnose_gpu_performance()
    
    # Configuration
    config = {
        'data_path': "processed_data/brats128_split/",
        'output_path': "/tmp/output/",
        'batch_size': 1,
        'num_workers': 0,
        'epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'use_mixed_precision': True,
        'gradient_accumulation_steps': 4,
        'resume_from': None
    }
    
    # Train model
    model = train_model(**config)