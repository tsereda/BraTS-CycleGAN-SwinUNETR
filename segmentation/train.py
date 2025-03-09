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
import argparse

# Import the fixed loss and model
from losses import CombinedLoss
from swin_unetr import ImprovedSwinUNETR
from dataset import get_data_loaders, BraTSDataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BraTS Segmentation Training')
    parser.add_argument('--data_path', type=str, default="processed_data/brats128_split/",
                        help='Path to dataset')
    parser.add_argument('--output_path', type=str, default="/tmp/output/",
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
        feature_size=32  # Reduced from 48 to 32 to save memory
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
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False
    )
    
    # Learning rate scheduler - use cosine annealing for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate * 0.01
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
                if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                    current_memory = torch.cuda.memory_allocated(0) / 1e9
                    max_memory = torch.cuda.max_memory_allocated(0) / 1e9
                    
                    print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {batch_loss:.4f}, "
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
        
        # Calculate epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
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


def diagnose_gpu_performance():
    """Run more comprehensive tests to diagnose GPU performance issues"""
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot diagnose GPU performance.")
        return
    
    print("\n=== GPU Performance Diagnostics ===")
    
    # Basic info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Enable tensor core optimizations for A100
    if hasattr(torch.cuda, 'get_device_capability'):
        capability = torch.cuda.get_device_capability(0)
        print(f"CUDA Compute Capability: {capability[0]}.{capability[1]}")
        if capability[0] >= 8:  # A100 has compute capability 8.0
            torch.set_float32_matmul_precision('high')
            print("Enabled Tensor Core optimizations")
    
    # Test matrix multiplication (CPU vs GPU) with various sizes
    sizes = [1000, 2000, 4000, 8192]  # Added larger size for A100
    
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
        
        # Test with default precision
        gpu_start = time.time()
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_end = time.time()
        gpu_time = gpu_end - gpu_start
        
        # Test with mixed precision (FP16) if on A100
        if hasattr(torch.cuda, 'get_device_capability') and torch.cuda.get_device_capability(0)[0] >= 8:
            with torch.amp.autocast(device_type='cuda'):
                torch.cuda.synchronize()
                gpu_fp16_start = time.time()
                _ = torch.matmul(a_gpu, b_gpu)
                torch.cuda.synchronize()
                gpu_fp16_end = time.time()
            gpu_fp16_time = gpu_fp16_end - gpu_fp16_start
            
            print(f"  CPU Time: {cpu_time:.4f}s")
            print(f"  GPU Time (FP32): {gpu_time:.4f}s")
            print(f"  GPU Time (FP16): {gpu_fp16_time:.4f}s")
            print(f"  Speedup vs CPU (FP32): {cpu_time / gpu_time:.1f}x")
            print(f"  Speedup vs CPU (FP16): {cpu_time / gpu_fp16_time:.1f}x")
            print(f"  FP16 vs FP32 Speedup: {gpu_time / gpu_fp16_time:.1f}x")
        else:
            speedup = cpu_time / gpu_time
            print(f"  CPU Time: {cpu_time:.4f}s")
            print(f"  GPU Time: {gpu_time:.4f}s")
            print(f"  Speedup: {speedup:.1f}x")
        
        # Clean up
        del a_cpu, b_cpu, a_gpu, b_gpu, _
        torch.cuda.empty_cache()
    
    # Test memory bandwidth
    print("\nTesting memory bandwidth:")
    tensor_size = 50000000  # 50M elements for A100 (was 10M)
    
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
    theoretical_max = 1935  # A100 has ~1935 GB/s theoretical bandwidth
    print(f"  Theoretical Max Bandwidth: ~{theoretical_max} GB/s")
    print(f"  Efficiency: {(bandwidth/theoretical_max)*100:.1f}%")
    
    # Clean up
    del x_gpu, y_gpu, _
    torch.cuda.empty_cache()
    
    # Test cuDNN benchmark mode impact
    print("\nTesting cuDNN benchmark mode impact:")
    
    # Create test tensors
    input_tensor = torch.randn(8, 64, 64, 64, 64, device='cuda')  # B, C, D, H, W
    conv3d = torch.nn.Conv3d(64, 64, kernel_size=3, padding=1).cuda()
    
    # Test without benchmark
    torch.backends.cudnn.benchmark = False
    
    # Warmup
    _ = conv3d(input_tensor)
    torch.cuda.synchronize()
    
    # Measure
    trials = 10
    start = time.time()
    for _ in range(trials):
        _ = conv3d(input_tensor)
    torch.cuda.synchronize()
    end = time.time()
    time_no_benchmark = (end - start) / trials
    
    # Test with benchmark
    torch.backends.cudnn.benchmark = True
    
    # Warmup again with benchmark enabled
    _ = conv3d(input_tensor)
    torch.cuda.synchronize()
    
    # Measure
    start = time.time()
    for _ in range(trials):
        _ = conv3d(input_tensor)
    torch.cuda.synchronize()
    end = time.time()
    time_with_benchmark = (end - start) / trials
    
    print(f"  Conv3D without benchmark: {time_no_benchmark:.4f}s")
    print(f"  Conv3D with benchmark: {time_with_benchmark:.4f}s")
    print(f"  Speedup from benchmark: {time_no_benchmark / time_with_benchmark:.2f}x")
    
    # Clean up
    del input_tensor, conv3d, _
    torch.cuda.empty_cache()
    
    print("\nGPU Performance Diagnostics Complete")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Run diagnostics first
    diagnose_gpu_performance()
    
    # Configuration for A100 GPU
    config = {
        'data_path': "processed_data/brats128_split/",
        'output_path': "/tmp/output/",
        'batch_size': 4,        # Increased batch size
        'num_workers': 4,       # Increased workers
        'epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'use_mixed_precision': True,
        'gradient_accumulation_steps': 1,  # Reduced for A100
        'resume_from': None,
        'benchmark_mode': True  # Enable cuDNN benchmarking
    }
    
    # Train model
    model = train_model(**config)