import os
import shutil
import time

# Set shared Kaggle cache directory BEFORE importing kagglehub
shared_cache_dir = "/home/santosh_lab/shared/kaggle_cache/.kagglehub"
os.environ['KAGGLEHUB_CACHE_DIR'] = shared_cache_dir

# Now import kagglehub
import kagglehub

# Verify the cache directory is the shared one
print(f"Using Kaggle cache directory: {os.environ.get('KAGGLEHUB_CACHE_DIR')}")

# Make sure the shared cache directory exists
os.makedirs(shared_cache_dir, exist_ok=True)

# Download the dataset
dataset_name = "awsaf49/brats20-dataset-training-validation"
print(f"Downloading dataset: {dataset_name}")

try:
    # Download the entire dataset
    path = kagglehub.dataset_download(dataset_name)
    print(f"Path to downloaded dataset: {path}")
    
    # Locate the training data directory within the downloaded dataset
    training_data_dir = os.path.join(path, "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
    if os.path.exists(training_data_dir):
        print(f"Found training data at: {training_data_dir}")
        source_dir = training_data_dir
        
    # Set destination directory for the training data
    raw_data_dir = "./brats20-dataset/"
    os.makedirs(raw_data_dir, exist_ok=True)
    print(f"Ensured raw_data directory exists: {raw_data_dir}")
    
    # Get list of items to copy
    items = os.listdir(source_dir)
    total_items = len(items)
    print(f"Found {total_items} items to copy")
    
    # Simple progress indicator
    start_time = time.time()
    for i, item in enumerate(items, 1):
        source_path = os.path.join(source_dir, item)
        destination_path = os.path.join(raw_data_dir, item)
        
        # Simple progress indicator
        percent_done = (i / total_items) * 100
        elapsed = time.time() - start_time
        print(f"[{i}/{total_items}] {percent_done:.1f}% - Copying: {item}")
        
        # Check if the item is a directory
        if os.path.isdir(source_path):
            # If destination already exists, remove it
            if os.path.exists(destination_path):
                shutil.rmtree(destination_path)
            # Copy the directory and its contents
            shutil.copytree(source_path, destination_path)
            print(f"✓ Copied directory: {item}")
        else:
            # Copy the file
            shutil.copy2(source_path, destination_path)
            print(f"✓ Copied file: {item}")
    
    total_time = time.time() - start_time
    print(f"All {total_items} items copied to {raw_data_dir} in {total_time:.1f} seconds")
    
except Exception as e:
    print(f"Error: {e}")
    print("Please check your Kaggle API credentials and internet connection.")