import os
import glob
import json
from pathlib import Path
import argparse


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


def count_split_dataset(base_path):
    """
    Count the number of cases in a split dataset (train/val)
    """
    print(f"\n=== Counting cases in split dataset {base_path} ===")
    
    # Check if the base path exists
    if not os.path.exists(base_path):
        print(f"ERROR: Path {base_path} does not exist!")
        return {"train": 0, "val": 0, "total": 0}
    
    # Count train cases
    train_path = os.path.join(base_path, "train", "images")
    if os.path.exists(train_path):
        train_files = glob.glob(os.path.join(train_path, "*.npy"))
        train_count = len(train_files)
        
        # Extract unique case IDs
        train_case_ids = set()
        for img_file in train_files:
            case_id = os.path.basename(img_file).replace("image_", "").replace(".npy", "")
            train_case_ids.add(case_id)
            
        print(f"Found {len(train_case_ids)} unique training cases")
    else:
        print(f"ERROR: Training images directory not found at {train_path}")
        train_case_ids = set()
    
    # Count validation cases
    val_path = os.path.join(base_path, "val", "images")
    if os.path.exists(val_path):
        val_files = glob.glob(os.path.join(val_path, "*.npy"))
        val_count = len(val_files)
        
        # Extract unique case IDs
        val_case_ids = set()
        for img_file in val_files:
            case_id = os.path.basename(img_file).replace("image_", "").replace(".npy", "")
            val_case_ids.add(case_id)
            
        print(f"Found {len(val_case_ids)} unique validation cases")
    else:
        print(f"ERROR: Validation images directory not found at {val_path}")
        val_case_ids = set()
    
    # Check if there's any overlap (there shouldn't be)
    overlap = train_case_ids.intersection(val_case_ids)
    if overlap:
        print(f"WARNING: Found {len(overlap)} cases that appear in both train and validation sets!")
    
    total_cases = len(train_case_ids) + len(val_case_ids)
    print(f"Total unique cases in split dataset: {total_cases}")
    
    return {
        "train": len(train_case_ids), 
        "val": len(val_case_ids), 
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
    pattern = "BraTS20_*"
    patient_dirs = glob.glob(os.path.join(base_path, pattern))
    
    # If no directories found with this pattern, try searching subdirectories
    if not patient_dirs:
        print(f"No directories found with pattern '{pattern}'. Trying to search in subdirectories...")
        patient_dirs = glob.glob(os.path.join(base_path, "**", pattern), recursive=True)
    
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


def check_processing_results(processed_path, dataset_type="training"):
    """
    Check the processing_results_{dataset_type}.json file if it exists
    """
    results_file = os.path.join(processed_path, f"processing_results_{dataset_type}.json")
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
    else:
        print(f"Processing results file not found: {results_file}")
    
    return None


def main():
    """
    Main function to count all datasets
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Count BraTS2020 dataset cases in various processing stages')
    
    # Input paths with defaults pointing to /data directories
    parser.add_argument('--input_train', type=str, default='/data/BraTS2020_TrainingData', 
                        help='Path to raw training data (default: /data/BraTS2020_TrainingData)')
    parser.add_argument('--input_val', type=str, default='/data/BraTS2020_ValidationData', 
                        help='Path to raw validation data (default: /data/BraTS2020_ValidationData)')
    
    # Output base path
    parser.add_argument('--output_base', type=str, default='/data/processed',
                        help='Base directory for all output folders (default: /data/processed)')
    
    args = parser.parse_args()
    
    # Create derived output paths based on the base output directory
    output_base = Path(args.output_base)
    
    # Use the same path structure as in the preprocessing script
    PROCESSED_TRAINING_PATH = str(output_base / 'brats128_training')
    PROCESSED_VALIDATION_PATH = str(output_base / 'brats128_validation')
    SPLIT_DATA_PATH = str(output_base / 'brats128_split')
    CYCLEGAN_DATA_PATH = str(output_base / 'brats128_cyclegan')
    
    # Print the paths being used
    print("\n=== PATHS BEING USED ===")
    print(f"Raw Training Data: {args.input_train}")
    print(f"Raw Validation Data: {args.input_val}")
    print(f"Processed Training Data: {PROCESSED_TRAINING_PATH}")
    print(f"Processed Validation Data: {PROCESSED_VALIDATION_PATH}")
    print(f"Split Dataset: {SPLIT_DATA_PATH}")
    print(f"CycleGAN Dataset: {CYCLEGAN_DATA_PATH}")
    
    # Count raw datasets
    print("\n=== COUNTING RAW DATASETS ===")
    raw_training_count = count_raw_dataset(args.input_train)
    raw_validation_count = count_raw_dataset(args.input_val)
    
    # Count processed datasets
    print("\n=== COUNTING PROCESSED DATASETS ===")
    processed_training_count = count_dataset_cases(PROCESSED_TRAINING_PATH)
    processed_validation_count = count_dataset_cases(PROCESSED_VALIDATION_PATH)
    
    # Check processing results if available
    training_results = check_processing_results(PROCESSED_TRAINING_PATH, "training")
    validation_results = check_processing_results(PROCESSED_VALIDATION_PATH, "validation")
    
    # Count split dataset
    split_counts = count_split_dataset(SPLIT_DATA_PATH)
    
    # Count CycleGAN dataset
    cyclegan_count = count_dataset_cases(CYCLEGAN_DATA_PATH)
    
    # Print summary
    print("\n=== DATASET COUNT SUMMARY ===")
    print(f"Raw Training Cases: {raw_training_count}")
    print(f"Raw Validation Cases: {raw_validation_count}")
    print(f"Processed Training Cases: {processed_training_count}")
    print(f"Processed Validation Cases: {processed_validation_count}")
    print(f"Split Dataset - Training Cases: {split_counts['train']}")
    print(f"Split Dataset - Validation Cases: {split_counts['val']}")
    print(f"CycleGAN Dataset Cases: {cyclegan_count}")
    
    # Calculate total cases for CycleGAN (should be train + validation)
    expected_cyclegan = split_counts['train'] + processed_validation_count
    if cyclegan_count != expected_cyclegan:
        print(f"WARNING: CycleGAN dataset count ({cyclegan_count}) doesn't match expected count ({expected_cyclegan})")
    else:
        print(f"CycleGAN dataset count matches expected count (train split + validation = {expected_cyclegan})")


if __name__ == "__main__":
    main()