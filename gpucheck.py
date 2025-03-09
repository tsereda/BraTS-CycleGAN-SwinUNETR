import torch
import time

def test_gpu_cuda_fixed():
    print("--- Improved PyTorch GPU/CUDA Test Script ---")

    # 1. Check CUDA Availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nCUDA is available! PyTorch is using GPU.")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    else:
        device = torch.device('cpu')
        print("\nCUDA is NOT available. PyTorch is using CPU.")
        return

    print(f"\nUsing device: {device}")

    # 2. Ensure cuDNN is enabled for better performance
    torch.backends.cudnn.benchmark = True
    print(f"cuDNN benchmark mode: {torch.backends.cudnn.benchmark}")

    # 3. Perform more reliable performance test
    size = 10000  # Larger test for better measurement
    iterations = 5
    
    # GPU Warmup - important for accurate testing
    warmup_tensor = torch.randn(size, size, device=device)
    warmup_result = torch.matmul(warmup_tensor, warmup_tensor)
    torch.cuda.synchronize()  # Wait for GPU to finish
    
    # GPU Performance with proper synchronization
    gpu_tensor = torch.randn(size, size, device=device)
    torch.cuda.synchronize()
    start_time_gpu = time.time()
    
    for _ in range(iterations):
        gpu_result = torch.matmul(gpu_tensor, gpu_tensor)
        torch.cuda.synchronize()  # Ensure operation is complete
    
    end_time_gpu = time.time()
    gpu_time = end_time_gpu - start_time_gpu

    # CPU Performance
    cpu_tensor = torch.randn(size, size)
    start_time_cpu = time.time()
    
    for _ in range(iterations):
        cpu_result = torch.matmul(cpu_tensor, cpu_tensor)
    
    end_time_cpu = time.time()
    cpu_time = end_time_cpu - start_time_cpu

    print(f"\nGPU Time ({iterations} iterations of {size}x{size} matrix multiplication): {gpu_time:.4f} seconds")
    print(f"CPU Time ({iterations} iterations of {size}x{size} matrix multiplication): {cpu_time:.4f} seconds")

    if device.type == 'cuda':
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"\nSpeedup (CPU/GPU): {speedup:.2f}x")
        
        if speedup < 1:
            print("\nWARNING: GPU is slower than CPU! This indicates a serious configuration issue.")
            print("Potential issues to check:")
            print("1. Ensure PyTorch is built with the correct CUDA version")
            print("2. Check for GPU thermal throttling or power limitations")
            print("3. Verify that no other processes are using the GPU")
            print("4. Try reinstalling PyTorch with the correct CUDA version for your system")
        elif speedup > 50:
            print("\nGPU performance is excellent!")
        elif speedup > 10:
            print("\nGPU performance is good.")
        else:
            print("\nGPU performance is modest but functional.")

    print("\n--- Test Script Completed ---")

if __name__ == "__main__":
    test_gpu_cuda_fixed()