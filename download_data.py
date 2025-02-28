import os
import shutil

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
dataset_name = "awsaf49/brats2020-training-data"
print(f"Downloading dataset: {dataset_name}")
path = kagglehub.dataset_download(dataset_name)
print(f"Path to dataset files: {path}")

# Set appropriate raw_data directory
raw_data_dir = "./brats2020-training-data/"
os.makedirs(raw_data_dir, exist_ok=True)
print(f"Ensured raw_data directory exists: {raw_data_dir}")

# Copy the files
for item in os.listdir(path):
    source_path = os.path.join(path, item)
    destination_path = os.path.join(raw_data_dir, item)
    
    # Check if the item is a directory
    if os.path.isdir(source_path):
        # If destination already exists, remove it
        if os.path.exists(destination_path):
            shutil.rmtree(destination_path)
        # Copy the directory and its contents
        shutil.copytree(source_path, destination_path)
        print(f"Copied directory: {item} to {raw_data_dir}")
    else:
        # Copy the file
        shutil.copy2(source_path, destination_path)
        print(f"Copied file: {item} to {raw_data_dir}")

print(f"All items copied to {raw_data_dir}")