"""Custom loss functions for LSTM regression."""

import torch
import torch.nn as nn


class PenalizedMSELoss(nn.Module):
    """MSE Loss with false positive penalty for demand response predictions."""
    
    def __init__(self, fp_penalty_weight=1.0, fp_penalty_threshold=0.05):
        """
        Initialize the penalized MSE loss.
        
        Args:
            fp_penalty_weight: Weight for the false positive penalty term
            fp_penalty_threshold: Threshold for considering a prediction as false positive
        """
        super(PenalizedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.fp_penalty_weight = fp_penalty_weight
        self.fp_penalty_threshold = fp_penalty_threshold
    
    def forward(self, predictions, targets):
        """
        Compute penalized MSE loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Tuple of (total_loss, base_loss, penalty_term)
        """
        # Base MSE loss
        base_loss = self.mse_loss(predictions, targets)
        
        # False positive penalty: penalize predictions > threshold when target is 0
        # This helps reduce false alarms for demand response events
        mask_zero_targets = (targets == 0).float()
        false_positives = torch.relu(predictions - self.fp_penalty_threshold) * mask_zero_targets
        penalty_term = self.fp_penalty_weight * torch.mean(false_positives ** 2)
        
        # Total loss
        total_loss = base_loss + penalty_term
        
        return total_loss, base_loss, penalty_term


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()