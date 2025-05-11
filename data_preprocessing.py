import os
import numpy as np
import nibabel as nib
from pathlib import Path
import json
import glob
import shutil
import subprocess
from typing import Tuple, List, Dict
import multiprocessing
from functools import partial
import time
import sys
import argparse
from tqdm import tqdm
import random
import stat  # For permission constants


def check_and_create_directories(paths_list, permission_mode=0o755):
    """
    Check and create directories with specific permissions
    
    Args:
        paths_list (list): List of directory paths to check/create
        permission_mode (int): Permission mode to set (default: 0o755 - rwxr-xr-x)
        
    Returns:
        dict: Status of each directory {'path': {'exists': bool, 'writable': bool}}
    """
    results = {}
    
    for path in paths_list:
        path_obj = Path(path)
        status = {'exists': False, 'writable': False}
        
        # Check if directory exists
        if path_obj.exists():
            status['exists'] = True
            
            # Check if directory is writable
            if os.access(path, os.W_OK):
                status['writable'] = True
            
        # Create directory if it doesn't exist
        else:
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                os.chmod(path, permission_mode)
                status['exists'] = True
                status['writable'] = True
                print(f"Created directory: {path}")
            except Exception as e:
                print(f"Error creating directory {path}: {str(e)}")
        
        results[path] = status
    
    return results


def process_single_case(case_data, output_path, min_label_ratio=0.007, has_mask=True, crop_margin=35): 
    """
    Process a single case with optimized operations
    
    Args:
        case_data (tuple): Tuple containing (case_idx, flair_path, t1ce_path, t2_path, t1_path, mask_path)
                          If has_mask=False, mask_path can be None
        output_path (str): Path to save preprocessed data
        min_label_ratio (float): Minimum ratio of non-zero labels required
        has_mask (bool): Whether the case has a segmentation mask
        crop_margin (int): Margin to use for cropping (smaller value = less aggressive crop)
        
    Returns:
        tuple: (status, case_id) where status is True if valid, False if error, None if skipped
    """
    if has_mask:
        case_idx, flair_path, t1ce_path, t2_path, t1_path, mask_path = case_data
    else:
        case_idx, flair_path, t1ce_path, t2_path, t1_path = case_data
        mask_path = None
    
    # Extract case_id from directory name
    case_id = os.path.basename(os.path.dirname(t2_path))
    
    try:
        # For better output from multiple processes
        sys.stdout.write(f"Starting to process {case_id}...\n")
        sys.stdout.flush()
        
        # Load modalities and explicitly convert to the right types
        temp_image_flair = nib.load(flair_path).get_fdata()
        temp_image_t1ce = nib.load(t1ce_path).get_fdata()
        temp_image_t2 = nib.load(t2_path).get_fdata()
        temp_image_t1 = nib.load(t1_path).get_fdata()
        
        # Load mask if available
        if has_mask:
            temp_mask = nib.load(mask_path).get_fdata()
            # Convert mask to uint8 for memory efficiency
            temp_mask = temp_mask.astype(np.uint8)
            # Remap label 4 to 3 (following BraTS convention)
            temp_mask[temp_mask == 4] = 3
        
        # Explicitly convert to float32 (important for in-place operations)
        temp_image_flair = temp_image_flair.astype(np.float32)
        temp_image_t1ce = temp_image_t1ce.astype(np.float32)
        temp_image_t2 = temp_image_t2.astype(np.float32)
        temp_image_t1 = temp_image_t1.astype(np.float32)
        
        # Get dimensions of the original image
        orig_dims = temp_image_flair.shape
        
        # Calculate crop boundaries with asymmetric cropping (no cropping on right side)
        x_start = max(0, crop_margin)
        x_end = min(orig_dims[0] - crop_margin, 240 - crop_margin)  # Cropping x end (bottom)
        y_start = max(0, crop_margin)
        y_end = min(orig_dims[1] - crop_margin // 3, 240 - crop_margin // 3)  # Added cropping on right side
        z_start = max(0, crop_margin // 2)  # Less margin for z-axis
        z_end = min(orig_dims[2] - (crop_margin // 2), 155 - (crop_margin // 2))
        
        # Pre-crop to reduce memory footprint before normalization - less aggressive crop
        temp_image_flair = temp_image_flair[x_start:x_end, y_start:y_end, z_start:z_end]
        temp_image_t1ce = temp_image_t1ce[x_start:x_end, y_start:y_end, z_start:z_end]
        temp_image_t2 = temp_image_t2[x_start:x_end, y_start:y_end, z_start:z_end]
        temp_image_t1 = temp_image_t1[x_start:x_end, y_start:y_end, z_start:z_end]

        if has_mask:
            temp_mask = temp_mask[x_start:x_end, y_start:y_end, z_start:z_end]
            
            # Check if case has enough non-zero labels early to avoid unnecessary processing
            val, counts = np.unique(temp_mask, return_counts=True)
            
            if (1 - (counts[0]/counts.sum())) <= min_label_ratio:
                sys.stdout.write(f"Case {case_id} skipped: insufficient non-zero labels\n")
                sys.stdout.flush()
                return None, case_id  # Not enough non-zero labels
        
        # Optimize normalization using vectorized operations
        # Use in-place operations to reduce memory usage
        for img in [temp_image_flair, temp_image_t1ce, temp_image_t2, temp_image_t1]:
            # Verify data type to prevent errors
            if not np.issubdtype(img.dtype, np.floating):
                img = img.astype(np.float32)
                
            min_val = np.min(img)
            max_val = np.max(img)
            if max_val > min_val:  # Avoid division by zero
                img -= min_val
                img /= (max_val - min_val)
        
        # Stack modalities (flair, t1ce, t2, t1) - more memory efficient than separate operations
        temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2, temp_image_t1], axis=3)
        
        # Save the processed files
        np.save(
            Path(output_path) / 'images' / f'image_{case_id}.npy',
            temp_combined_images
        )
        
        if has_mask:
            np.save(
                Path(output_path) / 'masks' / f'mask_{case_id}.npy',
                temp_mask
            )
        
        sys.stdout.write(f"Case {case_id} processed successfully\n")
        sys.stdout.flush()
        return True, case_id
        
    except Exception as e:
        sys.stdout.write(f"Error processing {case_id}: {str(e)}\n")
        sys.stdout.flush()
        return False, case_id


def preprocess_brats2020(input_path: str, output_path: str, dataset_type: str = "training", has_mask: bool = True, crop_margin: int = 35):
    """
    Preprocess BraTS2020 dataset with parallel processing for speed
    
    Args:
        input_path (str): Path to raw BraTS2020 dataset (BraTS20_Training_* or BraTS20_Validation_*)
        output_path (str): Path to save preprocessed data
        dataset_type (str): Either "training" or "validation"
        has_mask (bool): Whether the dataset has segmentation masks
        crop_margin (int): Margin to use for cropping (smaller value = less aggressive crop)
    """
    print(f"Starting preprocessing of BraTS2020 {dataset_type} dataset...")
    print(f"Input path: {input_path}")
    print(f"Crop margin: {crop_margin} (smaller = less cropping)")
    
    # Determine number of workers (use 1 less than CPU count to avoid system freeze)
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Create output directories
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'images').mkdir(exist_ok=True)
    
    if has_mask:
        (output_path / 'masks').mkdir(exist_ok=True)
    
    # Find all directories
    print("Scanning for input files...")
    
    # Determine directory pattern based on dataset type
    dir_pattern = f'BraTS20_{dataset_type.capitalize()}_*' if dataset_type.lower() in ['training', 'validation'] else 'BraTS20_*'
    
    patient_dirs = sorted(glob.glob(f'{input_path}/{dir_pattern}'))
    
    # If no directories found with this pattern, try searching subdirectories
    if not patient_dirs:
        print(f"No directories found with pattern '{dir_pattern}'. Trying to search in subdirectories...")
        patient_dirs = sorted(glob.glob(f'{input_path}/**/{dir_pattern}', recursive=True))
    
    patient_dirs = [d for d in patient_dirs if os.path.isdir(d)]
    
    if not patient_dirs:
        print(f"ERROR: No patient directories found matching pattern '{dir_pattern}'")
        return {"valid_cases": [], "skipped_cases": []}
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    # Process all patient directories
    case_data = []
    
    for idx, patient_dir in enumerate(patient_dirs):
        patient_id = os.path.basename(patient_dir)
        
        # Define paths for each modality file
        flair_file = os.path.join(patient_dir, f"{patient_id}_flair.nii.gz")
        t1ce_file = os.path.join(patient_dir, f"{patient_id}_t1ce.nii.gz")
        t2_file = os.path.join(patient_dir, f"{patient_id}_t2.nii.gz")
        t1_file = os.path.join(patient_dir, f"{patient_id}_t1.nii.gz")
        
        if has_mask:
            mask_file = os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")
            
            # Check if all necessary files exist
            if all(os.path.exists(f) for f in [flair_file, t1ce_file, t2_file, t1_file, mask_file]):
                case_data.append((idx, flair_file, t1ce_file, t2_file, t1_file, mask_file))
            else:
                # Try alternative file extensions (.nii instead of .nii.gz)
                flair_file = os.path.join(patient_dir, f"{patient_id}_flair.nii")
                t1ce_file = os.path.join(patient_dir, f"{patient_id}_t1ce.nii")
                t2_file = os.path.join(patient_dir, f"{patient_id}_t2.nii")
                t1_file = os.path.join(patient_dir, f"{patient_id}_t1.nii")
                mask_file = os.path.join(patient_dir, f"{patient_id}_seg.nii")
                
                if all(os.path.exists(f) for f in [flair_file, t1ce_file, t2_file, t1_file, mask_file]):
                    case_data.append((idx, flair_file, t1ce_file, t2_file, t1_file, mask_file))
                else:
                    print(f"Warning: Missing files for {patient_id}. Skipping.")
        else:
            # For validation data without masks
            if all(os.path.exists(f) for f in [flair_file, t1ce_file, t2_file, t1_file]):
                case_data.append((idx, flair_file, t1ce_file, t2_file, t1_file))
            else:
                # Try alternative file extensions (.nii instead of .nii.gz)
                flair_file = os.path.join(patient_dir, f"{patient_id}_flair.nii")
                t1ce_file = os.path.join(patient_dir, f"{patient_id}_t1ce.nii")
                t2_file = os.path.join(patient_dir, f"{patient_id}_t2.nii")
                t1_file = os.path.join(patient_dir, f"{patient_id}_t1.nii")
                
                if all(os.path.exists(f) for f in [flair_file, t1ce_file, t2_file, t1_file]):
                    case_data.append((idx, flair_file, t1ce_file, t2_file, t1_file))
                else:
                    print(f"Warning: Missing files for {patient_id}. Skipping.")
    
    print(f"Found {len(case_data)} complete cases out of {len(patient_dirs)} directories")
    
    if len(case_data) == 0:
        print("ERROR: No valid cases found for processing!")
        return {"valid_cases": [], "skipped_cases": []}
    
    processed_files = {'valid_cases': [], 'skipped_cases': []}
    
    # Process cases
    print(f"Processing {len(case_data)} cases...")
    print(f"Using parallel processing with {num_workers} workers...")
    
    # Process first case separately to catch any setup issues early
    if case_data:
        print("Processing first case to check for issues...")
        first_status, first_case_id = process_single_case(case_data[0], output_path, has_mask=has_mask, crop_margin=crop_margin)
        
        if first_status is True:
            processed_files['valid_cases'].append(first_case_id)
        elif first_status is False:
            processed_files['skipped_cases'].append(first_case_id)
        
        # Process remaining cases in parallel
        remaining_case_data = case_data[1:]
        print(f"Processing remaining {len(remaining_case_data)} cases in parallel...")
        
        if remaining_case_data:
            # Set up the parallel processing function
            process_func = partial(process_single_case, output_path=output_path, has_mask=has_mask, crop_margin=crop_margin)
            
            # Process in smaller batches to avoid memory issues
            batch_size = min(32, len(remaining_case_data))
            batches = [remaining_case_data[i:i+batch_size] for i in range(0, len(remaining_case_data), batch_size)]
            
            total_processed = len(processed_files['valid_cases']) + len(processed_files['skipped_cases'])
            
            for batch_idx, batch in enumerate(batches):
                # Use a context manager to ensure proper cleanup of resources
                with multiprocessing.Pool(processes=num_workers) as pool:
                    batch_results = pool.map(process_func, batch)
                
                # Process batch results
                valid_in_batch = 0
                for status, case_id in batch_results:
                    if status is True:
                        processed_files['valid_cases'].append(case_id)
                        valid_in_batch += 1
                    elif status is False:
                        processed_files['skipped_cases'].append(case_id)
                
                total_processed = len(processed_files['valid_cases']) + len(processed_files['skipped_cases'])
                progress_percent = (total_processed / len(case_data)) * 100
                
                print(f"Batch {batch_idx+1}/{len(batches)} ({valid_in_batch}/{len(batch)} valid cases) - Overall: {total_processed}/{len(case_data)} ({progress_percent:.1f}%) processed")
    else:
        print("No cases found to process!")
    
    # Save processing results
    with open(output_path / f'processing_results_{dataset_type}.json', 'w') as f:
        json.dump(processed_files, f, indent=2)
    
    print(f"Preprocessing complete. Processed {len(processed_files['valid_cases'])} valid cases.")
    print(f"Skipped {len(processed_files['skipped_cases'])} cases.")
    
    return processed_files


def split_dataset_three_way(input_folder: str, output_folder: str, seg_ratio=0.4, cyclegan_ratio=0.4, test_ratio=0.2):
    """
    Split preprocessed dataset into three non-overlapping sets: segmentation, cyclegan, and test
    
    Args:
        input_folder (str): Path to preprocessed data
        output_folder (str): Path to save split data
        seg_ratio (float): Ratio for segmentation training
        cyclegan_ratio (float): Ratio for cyclegan training
        test_ratio (float): Ratio for test set
    """
    print(f"Splitting dataset with ratios: Segmentation={seg_ratio}, CycleGAN={cyclegan_ratio}, Test={test_ratio}")
    
    # Check if there are actually files to split
    image_files = glob.glob(f"{input_folder}/images/*.npy")
    mask_files = glob.glob(f"{input_folder}/masks/*.npy")
    
    image_count = len(image_files)
    mask_count = len(mask_files)
    
    if image_count == 0 or mask_count == 0:
        print(f"WARNING: No files found to split! Images: {image_count}, Masks: {mask_count}")
        return
    
    # Get case IDs from filenames
    case_ids = [os.path.basename(f).replace('image_', '').replace('.npy', '') for f in image_files]
    
    # Shuffle case IDs to randomize the split
    random.seed(42)  # For reproducibility
    random.shuffle(case_ids)
    
    # Calculate split sizes
    seg_count = int(image_count * seg_ratio)
    cyclegan_count = int(image_count * cyclegan_ratio)
    test_count = image_count - seg_count - cyclegan_count
    
    # Split case IDs
    seg_cases = case_ids[:seg_count]
    cyclegan_cases = case_ids[seg_count:seg_count+cyclegan_count]
    test_cases = case_ids[seg_count+cyclegan_count:]
    
    print(f"Split counts: Segmentation={len(seg_cases)}, CycleGAN={len(cyclegan_cases)}, Test={len(test_cases)}")
    
    # Create output directories
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create directories for each split with the new structure
    (output_path / 'segmentation' / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (output_path / 'segmentation' / 'train' / 'masks').mkdir(parents=True, exist_ok=True)
    (output_path / 'segmentation' / 'test' / 'images').mkdir(parents=True, exist_ok=True)
    (output_path / 'segmentation' / 'test' / 'masks').mkdir(parents=True, exist_ok=True)
    (output_path / 'cyclegan' / 'images').mkdir(parents=True, exist_ok=True)
    
    # Copy files to respective directories
    input_path = Path(input_folder)
    
    # Copy segmentation files
    print(f"Copying files for segmentation training dataset...")
    for case_id in seg_cases:
        # Copy image and mask for segmentation
        src_img = input_path / 'images' / f"image_{case_id}.npy"
        dst_img = output_path / 'segmentation' / 'train' / 'images' / f"image_{case_id}.npy"
        
        src_mask = input_path / 'masks' / f"mask_{case_id}.npy"
        dst_mask = output_path / 'segmentation' / 'train' / 'masks' / f"mask_{case_id}.npy"
        
        if os.path.exists(src_img) and os.path.exists(src_mask):
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)
    
    # Copy cyclegan files
    print(f"Copying files for CycleGAN dataset...")
    for case_id in cyclegan_cases:
        # Copy only image for cyclegan
        src_img = input_path / 'images' / f"image_{case_id}.npy"
        dst_img = output_path / 'cyclegan' / 'images' / f"image_{case_id}.npy"
        
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
    
    # Copy test files
    print(f"Copying files for segmentation test dataset...")
    for case_id in test_cases:
        # Copy image and mask for test set
        src_img = input_path / 'images' / f"image_{case_id}.npy"
        dst_img = output_path / 'segmentation' / 'test' / 'images' / f"image_{case_id}.npy"
        
        src_mask = input_path / 'masks' / f"mask_{case_id}.npy"
        dst_mask = output_path / 'segmentation' / 'test' / 'masks' / f"mask_{case_id}.npy"
        
        if os.path.exists(src_img) and os.path.exists(src_mask):
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)
    
    # Count files in each split
    seg_train_images = len(glob.glob(f"{output_folder}/segmentation/train/images/*.npy"))
    seg_train_masks = len(glob.glob(f"{output_folder}/segmentation/train/masks/*.npy"))
    seg_test_images = len(glob.glob(f"{output_folder}/segmentation/test/images/*.npy"))
    seg_test_masks = len(glob.glob(f"{output_folder}/segmentation/test/masks/*.npy"))
    cyclegan_images = len(glob.glob(f"{output_folder}/cyclegan/images/*.npy"))
    
    print(f"Dataset split complete:")
    print(f"  Segmentation Train: {seg_train_images} images, {seg_train_masks} masks")
    print(f"  Segmentation Test: {seg_test_images} images, {seg_test_masks} masks")
    print(f"  CycleGAN: {cyclegan_images} images")
    
    # Save case IDs for each split for reference
    split_info = {
        'segmentation_train': seg_cases,
        'segmentation_test': test_cases,
        'cyclegan': cyclegan_cases
    }
    
    with open(output_path / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    return split_info


def add_validation_to_cyclegan(cyclegan_path: str, validation_data_path: str):
    """
    Add validation images to an existing CycleGAN dataset
    
    Args:
        cyclegan_path (str): Path to existing CycleGAN images
        validation_data_path (str): Path to the processed validation data
    """
    print(f"Adding validation images to CycleGAN dataset...")
    
    # Get file lists
    existing_images = glob.glob(f"{cyclegan_path}/images/*.npy")
    val_images = glob.glob(f"{validation_data_path}/images/*.npy")
    
    print(f"Found {len(existing_images)} existing images and {len(val_images)} validation images")
    
    # Copy validation images
    print(f"Copying validation images...")
    for src in val_images:
        dest = os.path.join(cyclegan_path, 'images', os.path.basename(src))
        shutil.copy2(src, dest)
    
    # Count total files
    total_images = len(glob.glob(f"{cyclegan_path}/images/*.npy"))
    
    print(f"CycleGAN dataset update complete:")
    print(f"  Total images: {total_images} (added {len(val_images)} validation images)")

def create_complete_dataset(
    training_data_path: str,
    validation_data_path: str,
    raw_training_path: str,
    raw_validation_path: str,
    dataset_path: str,
    crop_margin: int = 35
):
    """
    Complete pipeline to prepare BraTS2020 dataset for both segmentation and CycleGAN training
    with non-overlapping data splits
    
    Args:
        training_data_path (str): Path to raw training data
        validation_data_path (str): Path to raw validation data
        raw_training_path (str): Path to save preprocessed training data
        raw_validation_path (str): Path to save preprocessed validation data
        dataset_path (str): Path to save the final dataset directory
        crop_margin (int): Margin to use for cropping (smaller value = less aggressive crop)
    """
    print(f"=== STARTING COMPLETE DATASET PREPARATION ===")
    print(f"Using crop margin: {crop_margin} (smaller = less cropping)")
    
    # Step 1: Preprocess the training dataset (with masks)
    print("\n=== STEP 1: PREPROCESSING TRAINING DATA ===")
    # Search for training directories
    training_dirs = sorted(glob.glob(f'{training_data_path}/BraTS20_Training_*'))
    if training_dirs:
        print(f"Found {len(training_dirs)} training directories")
        training_processed = preprocess_brats2020(
            training_data_path, 
            raw_training_path,
            dataset_type="training",
            has_mask=True,
            crop_margin=crop_margin
        )
        print(f"Training data processed: {len(training_processed['valid_cases'])} valid cases")
    else:
        print(f"WARNING: No training directories found. Skipping training data preprocessing.")
    
    # Step 2: Preprocess the validation dataset
    print("\n=== STEP 2: PREPROCESSING VALIDATION DATA ===")
    # Search for validation directories
    validation_dirs = sorted(glob.glob(f'{validation_data_path}/BraTS20_Validation_*'))
    
    if validation_dirs:
        print(f"Found {len(validation_dirs)} validation directories")
        # Check for the presence of mask files in the first validation directory
        first_dir = validation_dirs[0]
        dir_basename = os.path.basename(first_dir)
        mask_path_gz = os.path.join(first_dir, f"{dir_basename}_seg.nii.gz")
        mask_path = os.path.join(first_dir, f"{dir_basename}_seg.nii")
        
        has_mask_validation = os.path.exists(mask_path_gz) or os.path.exists(mask_path)
        print(f"Validation data has masks: {has_mask_validation}")
        
        validation_processed = preprocess_brats2020(
            validation_data_path, 
            raw_validation_path,
            dataset_type="validation",
            has_mask=has_mask_validation,
            crop_margin=crop_margin
        )
        print(f"Validation data processed: {len(validation_processed['valid_cases'])} valid cases")
    else:
        print(f"WARNING: No validation directories found. Skipping validation data preprocessing.")
    
    # Step 3: Split training data into three non-overlapping sets
    print("\n=== STEP 3: CREATING FINAL DATASET STRUCTURE ===")
    # Check if there's processed training data to split
    train_images = len(glob.glob(f"{raw_training_path}/images/*.npy"))
    train_masks = len(glob.glob(f"{raw_training_path}/masks/*.npy"))
    
    if train_images > 0 and train_masks > 0:
        print(f"Found {train_images} training images and {train_masks} masks to split")
        # Use 40% for segmentation, 40% for cyclegan, 20% for test
        split_info = split_dataset_three_way(
            raw_training_path, 
            dataset_path,
            seg_ratio=0.4,
            cyclegan_ratio=0.4,
            test_ratio=0.2
        )
    else:
        print(f"WARNING: No training data found to split. Skipping split step.")
    
    # Step 4: Add validation images to CycleGAN dataset
    print("\n=== STEP 4: ADDING VALIDATION IMAGES TO CYCLEGAN DATASET ===")
    # Check if there's training cyclegan and validation data
    train_cyclegan_images = len(glob.glob(f"{dataset_path}/cyclegan/images/*.npy"))
    val_images = len(glob.glob(f"{raw_validation_path}/images/*.npy"))

    if train_cyclegan_images > 0 and val_images > 0:
        print(f"Found {train_cyclegan_images} training cyclegan images and {val_images} validation images")
        add_validation_to_cyclegan(
            f"{dataset_path}/cyclegan",
            raw_validation_path
        )
    else:
        print(f"WARNING: No data found for CycleGAN dataset. Skipping validation image addition.")
    
    print(f"\n=== COMPLETE DATASET PREPARATION FINISHED ===")

    print(f"1. Segmentation training data: {dataset_path}/segmentation/train")
    print(f"2. Segmentation test data: {dataset_path}/segmentation/test")
    print(f"3. CycleGAN training data: {dataset_path}/cyclegan")
    
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process BraTS2020 dataset for segmentation and CycleGAN')
    
    # Input paths with defaults pointing to /data directories
    parser.add_argument('--input_train', type=str, default='/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData', 
                        help='Path to raw training data (default: /data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData)')
    parser.add_argument('--input_val', type=str, default='/data/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData', 
                        help='Path to raw validation data (default: /data/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData)')
    
    # Output paths
    parser.add_argument('--output_base', type=str, default='processed',
                        help='Base directory for all output folders (default: processed)')
    
    # Crop margin parameter
    parser.add_argument('--crop_margin', type=int, default=35,
                        help='Margin to use for cropping (smaller = less aggressive crop, default: 35)')
    
    args = parser.parse_args()
    
    # Create derived output paths based on the base output directory
    output_base = Path(args.output_base).expanduser()  # Expand ~ to home directory
    
    # Define all required directories with new naming scheme
    RAW_DIR = str(output_base / 'brats_raw')
    RAW_TRAINING_PATH = str(output_base / 'brats_raw' / 'training')
    RAW_VALIDATION_PATH = str(output_base / 'brats_raw' / 'validation')
    DATASET_PATH = str(output_base / 'brats_dataset')
    
    # Check and create all required directories
    print("=== CHECKING AND CREATING DIRECTORIES ===")
    all_dirs = [
        str(output_base),
        RAW_DIR,
        RAW_TRAINING_PATH,
        RAW_VALIDATION_PATH,
        DATASET_PATH,
        # Preprocessed raw data directories
        str(Path(RAW_TRAINING_PATH) / 'images'),
        str(Path(RAW_TRAINING_PATH) / 'masks'),
        str(Path(RAW_VALIDATION_PATH) / 'images'),
        str(Path(RAW_VALIDATION_PATH) / 'masks'),
        # Final dataset directories
        str(Path(DATASET_PATH) / 'segmentation'),
        str(Path(DATASET_PATH) / 'segmentation' / 'train' / 'images'),
        str(Path(DATASET_PATH) / 'segmentation' / 'train' / 'masks'),
        str(Path(DATASET_PATH) / 'segmentation' / 'test' / 'images'),
        str(Path(DATASET_PATH) / 'segmentation' / 'test' / 'masks'),
        str(Path(DATASET_PATH) / 'cyclegan' / 'images')
    ]
    
    dir_status = check_and_create_directories(all_dirs)
    
    # Check for any permission issues
    permission_issues = False
    for path, status in dir_status.items():
        if not status['exists'] or not status['writable']:
            print(f"WARNING: Directory {path} has issues - exists: {status['exists']}, writable: {status['writable']}")
            permission_issues = True
    
    if permission_issues:
        print("WARNING: Some directory permission issues detected. Script may fail.")
        sys.exit(1)  # Exit if there are permission issues
    else:
        print("All directories created and have proper permissions.")
    
    # Check read permissions for input directories
    input_dirs = [args.input_train, args.input_val]
    for input_dir in input_dirs:
        if os.path.exists(input_dir):
            if not os.access(input_dir, os.R_OK):
                print(f"WARNING: No read permission for input directory {input_dir}")
                permission_issues = True
        else:
            print(f"WARNING: Input directory {input_dir} does not exist")
            permission_issues = True
    
    if permission_issues:
        print("WARNING: Input directory issues detected. Script may fail.")
        sys.exit(1)  # Exit if there are permission issues
    
    start_time = time.time()
    
    create_complete_dataset(
        training_data_path=args.input_train,
        validation_data_path=args.input_val,
        raw_training_path=RAW_TRAINING_PATH,
        raw_validation_path=RAW_VALIDATION_PATH,
        dataset_path=DATASET_PATH,
        crop_margin=args.crop_margin
    )
    
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")