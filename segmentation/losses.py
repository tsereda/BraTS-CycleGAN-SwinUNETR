import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class segmentation
    """
    def __init__(self, weight=None, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight
        
    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape [B, C, D, H, W] where C is the number of classes
            targets: Tensor of shape [B, D, H, W] with class indices or [B, C, D, H, W] one-hot
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
            
        # Ensure shapes match for debugging
        if probs.shape != targets_one_hot.shape:
            print(f"Shape mismatch: probs {probs.shape}, targets_one_hot {targets_one_hot.shape}")
            raise ValueError(f"Shape mismatch in DiceLoss: {probs.shape} vs {targets_one_hot.shape}")
            
        # Flatten for dice calculation while keeping batch and class dimensions
        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
        
        # Calculate intersection and dice score
        numerator = 2 * torch.sum(probs_flat * targets_flat, dim=2) + self.smooth
        denominator = torch.sum(probs_flat, dim=2) + torch.sum(targets_flat, dim=2) + self.smooth
        
        # Calculate class-wise dice scores
        dice_scores = numerator / denominator
        
        # Apply class weights if provided
        if self.weight is not None:
            dice_scores = dice_scores * self.weight
            
        # Return mean dice loss (1 - dice score)
        return 1.0 - dice_scores.mean()


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance in segmentation with improved stability
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
        # Get probabilities from logits
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).permute(0, 4, 1, 2, 3).float()
        
        # Calculate focal loss
        # -alpha * (1 - pt)^gamma * log(pt)
        # where pt = p if y = 1, and pt = 1 - p if y = 0
        
        # Binary cross entropy
        bce = -targets_one_hot * torch.log(probs.clamp(min=1e-6)) - (1 - targets_one_hot) * torch.log((1 - probs).clamp(min=1e-6))
        
        # pt is the probability of the correct class
        pt = targets_one_hot * probs + (1 - targets_one_hot) * (1 - probs)
        
        # Focal term
        focal_term = (1 - pt) ** self.gamma
        
        # Combine with alpha
        loss = self.alpha * focal_term * bce
        
        # Apply class weights if provided
        if self.weights is not None:
            weights = self.weights.view(1, -1, 1, 1, 1).to(loss.device)
            loss = loss * weights
            
        # Return mean loss
        focal_loss = loss.mean()
        
        # Ensure loss is not negative
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
        
        # Print values for debugging
        print(f"Dice loss: {dice.item()}, Focal loss: {focal.item()}")
        
        # Handle any NaN values that might occur
        if torch.isnan(dice):
            print("Warning: NaN in Dice loss, using only Focal loss")
            return self.focal_weight * focal
            
        if torch.isnan(focal):
            print("Warning: NaN in Focal loss, using only Dice loss")
            return self.dice_weight * dice
            
        total_loss = self.dice_weight * dice + self.focal_weight * focal
        
        # Print before clamping
        print(f"Total loss before clamping: {total_loss.item()}")
        
        # Temporarily comment out clamping to see if it's causing issues
        # total_loss = torch.clamp(total_loss, min=0.0)
        
        return total_loss