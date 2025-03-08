import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable, Any
import random


class RandomFlip3D:
    """Randomly flip 3D volume along specified axes"""
    def __init__(self, axes=(0, 1, 2), p=0.5):
        self.axes = axes
        self.p = p
        
    def __call__(self, img, mask):
        for axis in self.axes:
            if random.random() < self.p:
                img = torch.flip(img, dims=[axis+1])  # +1 because dim 0 is channel
                mask = torch.flip(mask, dims=[axis])
        return img, mask


class RandomIntensityShift:
    """Randomly shift intensities in the image"""
    def __init__(self, shift_range=(-0.1, 0.1), p=0.3):
        self.shift_range = shift_range
        self.p = p
        
    def __call__(self, img, mask):
        if random.random() < self.p:
            shift = random.uniform(*self.shift_range)
            img = img + shift
            img = torch.clamp(img, 0, 1)  # Keep in valid range
        return img, mask


class RandomGaussianNoise:
    """Add random Gaussian noise to the image"""
    def __init__(self, std_range=(0, 0.05), p=0.2):
        self.std_range = std_range
        self.p = p
        
    def __call__(self, img, mask):
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            noise = torch.randn_like(img) * std
            img = img + noise
            img = torch.clamp(img, 0, 1)  # Keep in valid range
        return img, mask


class Compose:
    """Compose several transforms together"""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class BraTSDataset(Dataset):
    """Dataset for BraTS2020 data with augmentations"""
    def __init__(self, 
                data_dir: str, 
                mode: str = 'train',
                transform: Optional[Callable] = None,
                normalize: bool = True,
                debug: bool = False):
        """
        Initialize the dataset
        
        Args:
            data_dir (str): Base directory containing image and mask directories
            mode (str): 'train' or 'val'
            transform: Optional transform to apply
            normalize: Whether to normalize data to [0, 1]
            debug: Print debugging information
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        self.normalize = normalize
        self.debug = debug
        
        # Get file paths
        self.img_dir = self.data_dir / mode / 'images'
        self.mask_dir = self.data_dir / mode / 'masks'
        
        # Check if directories exist
        if not self.img_dir.exists():
            raise ValueError(f"Image directory not found: {self.img_dir}")
        if not self.mask_dir.exists():
            raise ValueError(f"Mask directory not found: {self.mask_dir}")
        
        self.img_files = sorted(list(self.img_dir.glob('*.npy')))
        self.mask_files = sorted(list(self.mask_dir.glob('*.npy')))
        
        # Sanity check - make sure we have matching numbers of images and masks
        assert len(self.img_files) == len(self.mask_files), \
            f"Number of images ({len(self.img_files)}) doesn't match number of masks ({len(self.mask_files)})"
        
        if self.debug:
            print(f"Found {len(self.img_files)} {mode} samples")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        try:
            # Load image and mask
            img_path = self.img_files[idx]
            mask_path = self.mask_files[idx]
            
            if self.debug:
                print(f"Loading: {img_path.name}, {mask_path.name}")
            
            # Load data
            img = np.load(img_path)
            mask = np.load(mask_path)
            
            # Debug information about data
            print(f"Image shape: {img.shape}, range: {img.min():.4f} to {img.max():.4f}")
            unique_mask_values = np.unique(mask)
            print(f"Mask shape: {mask.shape}, unique values: {unique_mask_values}")
        
            
            # Extract just the integer label from the mask (if one-hot encoded)
            if len(mask.shape) == 4 and mask.shape[3] in [3, 4]:  # One-hot encoded mask
                if self.debug:
                    print(f"Converting one-hot mask with shape {mask.shape} to class indices")
                mask = np.argmax(mask, axis=3)
            
            # Check for NaN values
            if np.isnan(img).any():
                if self.debug:
                    print(f"Warning: NaN values found in image. Replacing with zeros.")
                img = np.nan_to_num(img)
            
            # Normalize if needed
            if self.normalize:
                # Normalize each channel separately
                for c in range(img.shape[3]):
                    channel = img[:, :, :, c]
                    if channel.max() > channel.min():  # Avoid division by zero
                        img[:, :, :, c] = (channel - channel.min()) / (channel.max() - channel.min())
            
            # Convert to PyTorch tensors
            # Image: (D,H,W,C) -> (C,D,H,W)
            img = torch.from_numpy(img).float().permute(3, 0, 1, 2)
            mask = torch.from_numpy(mask).long()
            
            # Apply transforms if provided
            if self.transform is not None:
                img, mask = self.transform(img, mask)
            
            return img, mask
            
        except Exception as e:
            if self.debug:
                print(f"Error loading sample {idx} ({self.img_files[idx].name}): {e}")
            
            # Return a simple dummy sample instead of crashing
            dummy_img = torch.zeros((3, 128, 128, 128), dtype=torch.float32)
            dummy_mask = torch.zeros((128, 128, 128), dtype=torch.long)
            return dummy_img, dummy_mask


def get_transforms(mode: str) -> Optional[Callable]:
    """Get transforms for the specified mode"""
    if mode == 'train':
        return Compose([
            RandomFlip3D(axes=(0, 1, 2), p=0.5),
            RandomIntensityShift(shift_range=(-0.1, 0.1), p=0.3),
            RandomGaussianNoise(std_range=(0, 0.03), p=0.2)
        ])
    else:
        # No transforms for validation
        return None


def get_data_loaders(
    data_path: str,
    batch_size: int = 1,
    num_workers: int = 4,
    use_augmentation: bool = True,
    debug: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation"""
    if debug:
        print("-" * 30)
        print("DATA LOADER CONFIGURATION:")
        print(f"Data directory: {data_path}")
        print(f"Batch size: {batch_size}")
        print(f"Num workers: {num_workers}")
        print(f"Using augmentation: {use_augmentation}")
    
    # Get transforms
    train_transform = get_transforms('train') if use_augmentation else None
    val_transform = None  # No transforms for validation
    
    # Create datasets
    try:
        train_dataset = BraTSDataset(
            data_dir=data_path,
            mode='train',
            transform=train_transform,
            normalize=True,
            debug=debug
        )
    except Exception as e:
        print(f"Error creating training dataset: {e}")
        # Create empty dataset as fallback
        train_dataset = torch.utils.data.TensorDataset(
            torch.zeros((1, 3, 128, 128, 128)),
            torch.zeros((1, 128, 128, 128), dtype=torch.long)
        )

    try:
        val_dataset = BraTSDataset(
            data_dir=data_path,
            mode='val',
            transform=val_transform,
            normalize=True,
            debug=debug
        )
    except Exception as e:
        print(f"Error creating validation dataset: {e}")
        # Create empty dataset as fallback
        val_dataset = torch.utils.data.TensorDataset(
            torch.zeros((1, 3, 128, 128, 128)),
            torch.zeros((1, 128, 128, 128), dtype=torch.long)
        )
    
    if debug:
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    if debug:
        print(f"Train loader batches: {len(train_loader)}")
        print(f"Validation loader batches: {len(val_loader)}")
        print("-" * 30)
    
    return train_loader, val_loader