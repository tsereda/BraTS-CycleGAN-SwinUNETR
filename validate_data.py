import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from pathlib import Path

def count_dataset_cases(base_path):
    """
    Count the number of cases in a dataset directory
    """
    print(f"\n=== Counting cases in {base_path} ===")
    # Check if the base path exists
    if not os.path.exists(base_path):
        print(f"ERROR: Path {base_path} does not exist!")
        return 0
    
    # Count images
    image_path = os.path.join(base_path, "images")
    if os.path.exists(image_path):
        image_files = glob.glob(os.path.join(image_path, "*.npy"))
        image_count = len(image_files)
        print(f"Found {image_count} image files")
        
        # Extract unique case IDs from filenames
        case_ids = set()
        for img_file in image_files:
            # Extract case ID (e.g., BraTS20_Training_001 from image_BraTS20_Training_001.npy)
            case_id = os.path.basename(img_file).replace("image_", "").replace(".npy", "")
            case_ids.add(case_id)
        
        print(f"Found {len(case_ids)} unique cases")
        return len(case_ids)
    else:
        print(f"ERROR: Images directory not found at {image_path}")
        return 0

def count_segmentation_cyclegan(dataset_path):
    """
    Count the number of cases in segmentation and cyclegan directories
    """
    print(f"\n=== Counting cases in segmentation/cyclegan at {dataset_path} ===")
    # Check if the base path exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Path {dataset_path} does not exist!")
        return {"segmentation_train": 0, "segmentation_test": 0, "cyclegan": 0, "total": 0}
    
    # Count segmentation train cases
    seg_train_path = os.path.join(dataset_path, "segmentation", "train", "images")
    if os.path.exists(seg_train_path):
        seg_train_files = glob.glob(os.path.join(seg_train_path, "*.npy"))
        
        # Extract unique case IDs
        seg_train_case_ids = set()
        for img_file in seg_train_files:
            case_id = os.path.basename(img_file).replace("image_", "").replace(".npy", "")
            seg_train_case_ids.add(case_id)
        
        print(f"Found {len(seg_train_case_ids)} unique segmentation train cases")
    else:
        print(f"ERROR: Segmentation train images directory not found at {seg_train_path}")
        seg_train_case_ids = set()
    
    # Count segmentation test cases
    seg_test_path = os.path.join(dataset_path, "segmentation", "test", "images")
    if os.path.exists(seg_test_path):
        seg_test_files = glob.glob(os.path.join(seg_test_path, "*.npy"))
        
        # Extract unique case IDs
        seg_test_case_ids = set()
        for img_file in seg_test_files:
            case_id = os.path.basename(img_file).replace("image_", "").replace(".npy", "")
            seg_test_case_ids.add(case_id)
        
        print(f"Found {len(seg_test_case_ids)} unique segmentation test cases")
    else:
        print(f"ERROR: Segmentation test images directory not found at {seg_test_path}")
        seg_test_case_ids = set()
    
    # Count cyclegan cases
    cyclegan_path = os.path.join(dataset_path, "cyclegan", "images")
    if os.path.exists(cyclegan_path):
        cyclegan_files = glob.glob(os.path.join(cyclegan_path, "*.npy"))
        
        # Extract unique case IDs
        cyclegan_case_ids = set()
        for img_file in cyclegan_files:
            case_id = os.path.basename(img_file).replace("image_", "").replace(".npy", "")
            cyclegan_case_ids.add(case_id)
        
        print(f"Found {len(cyclegan_case_ids)} unique cyclegan cases")
    else:
        print(f"ERROR: CycleGAN images directory not found at {cyclegan_path}")
        cyclegan_case_ids = set()
    
    # Check for overlaps between sets (there shouldn't be any)
    seg_train_test_overlap = seg_train_case_ids.intersection(seg_test_case_ids)
    seg_train_cyclegan_overlap = seg_train_case_ids.intersection(cyclegan_case_ids)
    seg_test_cyclegan_overlap = seg_test_case_ids.intersection(cyclegan_case_ids)
    
    if seg_train_test_overlap:
        print(f"WARNING: Found {len(seg_train_test_overlap)} cases that appear in both segmentation train and test sets!")
    if seg_train_cyclegan_overlap:
        print(f"WARNING: Found {len(seg_train_cyclegan_overlap)} cases that appear in both segmentation train and cyclegan sets!")
    if seg_test_cyclegan_overlap:
        print(f"WARNING: Found {len(seg_test_cyclegan_overlap)} cases that appear in both segmentation test and cyclegan sets!")
    
    total_cases = len(seg_train_case_ids) + len(seg_test_case_ids) + len(cyclegan_case_ids)
    print(f"Total unique cases in dataset: {total_cases}")
    
    return {
        "segmentation_train": len(seg_train_case_ids),
        "segmentation_test": len(seg_test_case_ids),
        "cyclegan": len(cyclegan_case_ids),
        "total": total_cases
    }

def count_raw_dataset(base_path):
    """
    Count the number of cases in the raw dataset
    """
    print(f"\n=== Counting raw dataset cases in {base_path} ===")
    # Check if the path exists
    if not os.path.exists(base_path):
        print(f"ERROR: Path {base_path} does not exist!")
        return 0
    
    # Count patient directories
    if "Training" in base_path:
        patient_dirs = glob.glob(os.path.join(base_path, "BraTS20_Training_*"))
    elif "Validation" in base_path:
        patient_dirs = glob.glob(os.path.join(base_path, "BraTS20_Validation_*"))
    else:
        # Try both patterns
        patient_dirs = glob.glob(os.path.join(base_path, "BraTS20_Training_*")) + \
                      glob.glob(os.path.join(base_path, "BraTS20_Validation_*"))
    
    # Only count directories
    patient_dirs = [d for d in patient_dirs if os.path.isdir(d)]
    print(f"Found {len(patient_dirs)} patient directories")
    
    # Get the highest case number to estimate total expected cases
    max_case_num = 0
    for dir_path in patient_dirs:
        try:
            dir_name = os.path.basename(dir_path)
            case_num = int(dir_name.split('_')[-1])
            max_case_num = max(max_case_num, case_num)
        except:
            pass
    
    if max_case_num > 0:
        print(f"Highest case number found: {max_case_num}")
    
    return len(patient_dirs)

def check_processing_results(processed_path):
    """
    Check the processing_results.json file if it exists
    """
    results_file = os.path.join(processed_path, "processing_results_training.json")
    if not os.path.exists(results_file):
        results_file = os.path.join(processed_path, "processing_results.json")
    
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            valid_cases = len(results.get('valid_cases', []))
            skipped_cases = len(results.get('skipped_cases', []))
            
            print(f"\n=== Processing Results from {results_file} ===")
            print(f"Valid cases: {valid_cases}")
            print(f"Skipped cases: {skipped_cases}")
            print(f"Total processed: {valid_cases + skipped_cases}")
            
            return {"valid": valid_cases, "skipped": skipped_cases, "total": valid_cases + skipped_cases}
        except Exception as e:
            print(f"Error reading processing results: {str(e)}")
    return None

def visualize_brats_samples(data_dir, output_file, num_samples=3):
    """
    Create a single visualization of multiple BraTS samples and save as PNG
    Args:
        data_dir: Base directory containing processed data
        output_file: Path to save the output PNG
        num_samples: Number of samples to include
    """
    print(f"\n=== Creating visualization of {num_samples} segmentation samples ===")
    
    # Expand user directory paths
    data_dir = Path(data_dir).expanduser()
    output_file = Path(output_file).expanduser()
    
    # Get image and mask files from segmentation train directory
    img_path = data_dir / 'segmentation' / 'train' / 'images'
    mask_path = data_dir / 'segmentation' / 'train' / 'masks'
    
    img_files = sorted(glob.glob(str(img_path / '*.npy')))
    if not img_files:
        print(f"No image files found in {img_path}")
        return
    
    # Select random samples
    random.seed(42)  # For reproducibility
    selected_files = random.sample(img_files, min(num_samples, len(img_files)))
    
    # Create a large figure to hold all samples
    fig = plt.figure(figsize=(15, 5 * num_samples))
    fig.suptitle("BraTS Dataset - Segmentation Sample Visualization (Axial View)", fontsize=16)
    modality_names = ['FLAIR', 'T1CE', 'T2', 'T1']
    
    # Plot each sample
    for i, img_file in enumerate(selected_files):
        # Get corresponding mask file
        case_id = os.path.basename(img_file).replace('image_', '').replace('.npy', '')
        mask_file = mask_path / f"mask_{case_id}.npy"
        
        # Load data
        img_data = np.load(img_file)
        mask_data = np.load(mask_file) if os.path.exists(mask_file) else None
        
        # Get dimensions and middle slice for axial view
        x, y, z, modalities = img_data.shape
        mid_slice = z // 2
        
        # Row position for this sample (each sample gets 5 subplots - 4 modalities + 1 mask)
        row_pos = i * 5
        
        # Add sample ID
        plt.figtext(0.05, 0.98 - (i * 0.31), f"Sample {i+1}: {case_id} - Shape: {img_data.shape}",
                   fontsize=12, weight='bold')
        
        # Plot each modality
        for m in range(modalities):
            ax = plt.subplot(num_samples, 5, row_pos + m + 1)
            ax.imshow(img_data[:, :, mid_slice, m], cmap='gray')
            ax.set_title(f"{modality_names[m]}")
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Plot mask overlay if available
        if mask_data is not None:
            ax = plt.subplot(num_samples, 5, row_pos + 5)
            ax.imshow(img_data[:, :, mid_slice, 0], cmap='gray')  # FLAIR as background
            
            # Color the different segmentation classes
            mask_slice = mask_data[:, :, mid_slice]
            edema = np.ma.masked_where(mask_slice != 1, mask_slice)
            non_enhancing = np.ma.masked_where(mask_slice != 2, mask_slice)
            enhancing = np.ma.masked_where(mask_slice != 3, mask_slice)
            
            ax.imshow(edema, cmap='cool', alpha=0.5)
            ax.imshow(non_enhancing, cmap='autumn', alpha=0.5)
            ax.imshow(enhancing, cmap='winter', alpha=0.5)
            
            ax.set_title("Mask Overlay")
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for title
    
    # Save the figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    plt.close(fig)

def visualize_cyclegan_samples(data_dir, output_file, num_samples=3):
    """
    Create a visualization of CycleGAN samples (no masks)
    Args:
        data_dir: Base directory containing processed data
        output_file: Path to save the output PNG
        num_samples: Number of samples to include
    """
    print(f"\n=== Creating visualization of {num_samples} CycleGAN samples ===")
    
    # Expand user directory paths
    data_dir = Path(data_dir).expanduser()
    output_file = Path(output_file).expanduser()
    
    # Get image files from cyclegan dataset
    img_path = data_dir / 'cyclegan' / 'images'
    img_files = sorted(glob.glob(str(img_path / '*.npy')))
    
    if not img_files:
        print(f"No image files found in {img_path}")
        return
    
    # Select random samples
    random.seed(43)  # Different seed from segmentation
    selected_files = random.sample(img_files, min(num_samples, len(img_files)))
    
    # Create a figure to hold all samples
    fig = plt.figure(figsize=(15, 4 * num_samples))
    fig.suptitle("CycleGAN Dataset - Sample Visualization (Axial View)", fontsize=16)
    modality_names = ['FLAIR', 'T1CE', 'T2', 'T1']
    
    # Plot each sample
    for i, img_file in enumerate(selected_files):
        # Get case ID
        case_id = os.path.basename(img_file).replace('image_', '').replace('.npy', '')
        
        # Load data
        img_data = np.load(img_file)
        
        # Get dimensions and middle slice for axial view
        x, y, z, modalities = img_data.shape
        mid_slice = z // 2
        
        # Add sample ID
        plt.figtext(0.05, 0.98 - (i * 0.25), f"Sample {i+1}: {case_id} - Shape: {img_data.shape}",
                   fontsize=12, weight='bold')
        
        # Plot each modality
        for m in range(modalities):
            ax = plt.subplot(num_samples, 4, i * 4 + m + 1)
            ax.imshow(img_data[:, :, mid_slice, m], cmap='gray')
            ax.set_title(f"{modality_names[m]}")
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for title
    
    # Save the figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    plt.close(fig)

def main():
    """
    Main function to count all datasets and create visualizations
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Validate BraTS processed datasets')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Base directory for processed data (default: /app/BraTS-CycleGAN-SwinUNETR/processed/)')
    args = parser.parse_args()
    
    # Process base directory
    if args.data_dir:
        # Use provided data directory
        processed_base = os.path.abspath(args.data_dir)
    else:
        # Default to the project repository path
        processed_base = "/app/BraTS-CycleGAN-SwinUNETR/processed/"
    
    # Dataset paths with new structure
    RAW_DIR = os.path.join(processed_base, "brats_raw")
    RAW_TRAINING_PATH = os.path.join(RAW_DIR, "training")
    RAW_VALIDATION_PATH = os.path.join(RAW_DIR, "validation")
    DATASET_PATH = os.path.join(processed_base, "brats_dataset")
    
    # Count raw datasets if they exist
    print("\n=== COUNTING RAW PREPROCESSED DATASETS ===")
    raw_training_count = count_dataset_cases(RAW_TRAINING_PATH)
    raw_validation_count = count_dataset_cases(RAW_VALIDATION_PATH)
    
    # Check processing results if available
    training_results = check_processing_results(RAW_TRAINING_PATH)
    validation_results = check_processing_results(RAW_VALIDATION_PATH)
    
    # Count datasets
    dataset_counts = count_segmentation_cyclegan(DATASET_PATH)
    
    # Print summary
    print("\n=== DATASET COUNT SUMMARY ===")
    if raw_training_count > 0:
        print(f"Raw Preprocessed Training Cases: {raw_training_count}")
    if raw_validation_count > 0:
        print(f"Raw Preprocessed Validation Cases: {raw_validation_count}")
    
    print(f"Segmentation Train Cases: {dataset_counts['segmentation_train']}")
    print(f"Segmentation Test Cases: {dataset_counts['segmentation_test']}")
    print(f"CycleGAN Cases: {dataset_counts['cyclegan']}")
    
    # Calculate expected cyclegan count if raw validation exists
    if raw_validation_count > 0:
        # Theoretically cyclegan should include raw training cyclegan portion + validation
        # Assume 40% of training went to cyclegan as in the original script
        expected_cyclegan = int(raw_training_count * 0.4) + raw_validation_count
        if dataset_counts['cyclegan'] != expected_cyclegan:
            print(f"NOTE: CycleGAN dataset count ({dataset_counts['cyclegan']}) differs from theoretical count ({expected_cyclegan})")
            print(f" This may be due to filtering of validation cases during processing")
        else:
            print(f"CycleGAN dataset count matches expected count (training cyclegan portion + validation = {expected_cyclegan})")
    
    # Create visualizations
    print("\n=== CREATING VISUALIZATIONS ===")
    # Create a directory for visualizations
    vis_dir = os.path.join(processed_base, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize segmentation samples
    vis_path_seg = os.path.join(vis_dir, "segmentation_samples.png")
    visualize_brats_samples(DATASET_PATH, vis_path_seg, num_samples=3)
    
    # Visualize cyclegan samples
    vis_path_cycle = os.path.join(vis_dir, "cyclegan_samples.png")
    visualize_cyclegan_samples(DATASET_PATH, vis_path_cycle, num_samples=3)
    
    print(f"\nAll visualizations saved to: {vis_dir}")
    print(f" - Segmentation samples: {vis_path_seg}")
    print(f" - CycleGAN samples: {vis_path_cycle}")

if __name__ == "__main__":
    main()