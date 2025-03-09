import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Corrected Dice Loss for multi-class segmentation that guarantees positive values
    """
    def __init__(self, weight=None, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight
        
    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape [B, C, D, H, W] where C is the number of classes
            targets: Tensor of shape [B, D, H, W] with class indices
        """
        num_classes = logits.shape[1]
        
        # Get probabilities from logits
        probs = F.softmax(logits, dim=1)
        
        # Check if targets are already one-hot
        if targets.shape != probs.shape:
            # Convert targets to one-hot if needed
            targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes)
            # Rearrange to match shape [B, C, D, H, W]
            targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).contiguous()
        else:
            targets_one_hot = targets
        
        # Flatten for dice calculation while keeping batch and class dimensions
        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
        
        # Calculate intersection and union for each class
        intersection = torch.sum(probs_flat * targets_flat, dim=2)  # [B, C]
        union = torch.sum(probs_flat, dim=2) + torch.sum(targets_flat, dim=2)  # [B, C]
        
        # Calculate Dice score (add smooth to both numerator and denominator)
        dice_scores = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B, C]
        
        # Apply class weights if provided
        if self.weight is not None:
            weight = self.weight.to(dice_scores.device)
            dice_scores = dice_scores * weight
            
        # Average over classes and batches
        dice_loss = 1.0 - torch.mean(dice_scores)
        
        # Ensure loss is non-negative (should always be the case with this implementation)
        dice_loss = torch.clamp(dice_loss, min=0.0)
        
        return dice_loss


class FocalLoss(nn.Module):
    """
    Improved Focal loss for handling class imbalance in segmentation
    """
    def __init__(self, 
                 alpha: float = 0.25, 
                 gamma: float = 2.0,
                 weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Get probabilities from logits with epsilon for numerical stability
        probs = F.softmax(logits, dim=1) + 1e-10
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).permute(0, 4, 1, 2, 3).float()
        
        # Calculate pt (probability of correct class)
        pt = torch.sum(targets_one_hot * probs, dim=1)
        
        # Compute focal loss (using log for numerical stability)
        focal_term = (1 - pt) ** self.gamma
        focal_loss = -self.alpha * focal_term * torch.log(pt + 1e-10)
        
        # Average over all dimensions except batch
        focal_loss = focal_loss.mean(dim=(1, 2, 3))
        
        # Apply class weights if provided
        if self.weights is not None:
            # Apply weights to each batch element by class
            class_indices = targets.view(targets.size(0), -1)
            batch_weights = torch.zeros_like(focal_loss)
            
            # Compute weighted average for each batch element
            for b in range(targets.size(0)):
                batch_weights[b] = torch.mean(self.weights[class_indices[b]])
                
            focal_loss = focal_loss * batch_weights
        
        # Return mean loss over batch
        focal_loss = focal_loss.mean()
        
        # Ensure loss is non-negative
        focal_loss = torch.clamp(focal_loss, min=0.0)
        
        return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined Dice and Focal loss with improved stability
    """
    def __init__(self, 
                 dice_weight: float = 1.0,
                 focal_weight: float = 1.0,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(weight=class_weights)
        self.focal_loss = FocalLoss(weights=class_weights)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        
        # Combine losses
        total_loss = self.dice_weight * dice + self.focal_weight * focal
        
        # Print diagnostic information
        #print(f"Dice loss: {dice.item()}, Focal loss: {focal.item()}")
        #print(f"Total loss before clamping: {total_loss.item()}")
        #print(f"Raw loss: {total_loss.item()}, min: {logits.min().item()}, max: {logits.max().item()}")
        
        # Ensure the total loss is not negative
        total_loss = torch.clamp(total_loss, min=0.0)
        
        return total_loss