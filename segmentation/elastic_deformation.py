import torch
import torch.nn.functional as F
import random
import numpy as np

class SimplifiedElasticDeformation:
    """
    Simplified elastic deformation for 3D volumes that avoids using custom Gaussian blur
    Instead uses a simple approach with smoother displacement fields
    """
    def __init__(self, alpha=15.0, p=0.3):
        self.alpha = alpha  # Controls deformation strength
        self.p = p  # Probability of applying deformation
        
    def __call__(self, img, mask):
        if random.random() >= self.p:
            return img, mask
        
        try:
            # Get shape
            shape = mask.shape  # [D, H, W]
            
            # Create a simpler displacement field (without trying to do Gaussian blur)
            # We'll use a coarse grid and then interpolate to make it smooth
            
            # 1. Create a coarse displacement grid (1/4 the size)
            coarse_shape = [max(s // 4, 1) for s in shape]
            coarse_displ = torch.randn(3, *coarse_shape, device=img.device) * self.alpha / 10.0
            
            # 2. Upsample to original size to get smoother fields
            displacement = F.interpolate(
                coarse_displ.unsqueeze(0),  # Add batch dim [1, 3, D', H', W']
                size=shape,                 # Target size [D, H, W]
                mode='trilinear',           # Trilinear interpolation for smoothness
                align_corners=False
            ).squeeze(0)  # Remove batch dim [3, D, H, W]
            
            # Create sampling grid
            grid_x, grid_y, grid_z = torch.meshgrid(
                torch.linspace(-1, 1, shape[0]),
                torch.linspace(-1, 1, shape[1]),
                torch.linspace(-1, 1, shape[2]),
                indexing='ij'
            )
            grid = torch.stack([grid_z, grid_y, grid_x], dim=3)  # (D, H, W, 3)
            
            # Add displacement to grid (ensuring displacement is properly scaled)
            grid = grid + displacement.permute(1, 2, 3, 0) * 0.05  # Scale factor to prevent extreme deformations
            
            # Apply deformation
            img = F.grid_sample(img.unsqueeze(0), grid.unsqueeze(0), mode='bilinear', 
                              padding_mode='border', align_corners=False).squeeze(0)
            mask = F.grid_sample(mask.float().unsqueeze(0).unsqueeze(0), grid.unsqueeze(0), 
                               mode='nearest', padding_mode='border', align_corners=False).squeeze(0).squeeze(0).long()
            
        except Exception as e:
            # If any error occurs, just return the original
            print(f"Warning: Error in simplified elastic deformation: {e}. Skipping augmentation.")
        
        return img, mask