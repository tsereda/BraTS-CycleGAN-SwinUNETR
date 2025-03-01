import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Optimized Dice loss for 3D segmentation with improved stability
    """
    def __init__(self, 
                 weights: Optional[torch.Tensor] = None, 
                 smooth: float = 1e-5,
                 square_denominator: bool = False):
        super().__init__()
        self.weights = weights
        self.smooth = smooth
        self.square_denominator = square_denominator
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Get probabilities from logits
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets: (B, C, D, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).permute(0, 4, 1, 2, 3).float()
        
        # Flatten predictions and targets
        probs = probs.view(probs.size(0), probs.size(1), -1)
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
        
        # Calculate Dice coefficient for each class
        # numerator: 2 * sum(pred * target) + smooth
        # denominator: sum(pred) + sum(target) + smooth
        numerator = 2 * torch.sum(probs * targets_one_hot, dim=2) + self.smooth
        
        if self.square_denominator:
            denominator = torch.sum(probs ** 2, dim=2) + torch.sum(targets_one_hot ** 2, dim=2) + self.smooth
        else:
            denominator = torch.sum(probs, dim=2) + torch.sum(targets_one_hot, dim=2) + self.smooth
            
        dice_per_class = numerator / denominator  # Shape: (B, C)
        
        # Apply class weights if provided
        if self.weights is not None:
            weights = self.weights.to(dice_per_class.device)
            dice_per_class = dice_per_class * weights
            
        # Calculate mean Dice over classes
        dice_loss = 1.0 - dice_per_class.mean()
        
        # Ensure loss is not negative
        dice_loss = torch.clamp(dice_loss, min=0.0)
            
        return dice_loss


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
        self.dice_loss = DiceLoss(weights=class_weights)
        self.focal_loss = FocalLoss(weights=class_weights)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        
        # Handle any NaN values that might occur
        if torch.isnan(dice):
            print("Warning: NaN in Dice loss, using only Focal loss")
            return self.focal_weight * focal
            
        if torch.isnan(focal):
            print("Warning: NaN in Focal loss, using only Dice loss")
            return self.dice_weight * dice
            
        total_loss = self.dice_weight * dice + self.focal_weight * focal
        
        # Ensure the total loss is not negative
        total_loss = torch.clamp(total_loss, min=0.0)
        
        return total_loss