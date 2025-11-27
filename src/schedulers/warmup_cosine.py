"""Learning rate schedulers."""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.
    
    During warmup, learning rate increases linearly from 0 to base_lr.
    After warmup, learning rate follows cosine annealing schedule.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            warmup_epochs: Number of epochs for warmup
            total_epochs: Total number of training epochs
            min_lr: Minimum learning rate
            last_epoch: The index of last epoch
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


def create_scheduler(optimizer: Optimizer, config: dict) -> WarmupCosineAnnealingLR:
    """
    Factory function to create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to wrap
        config: Configuration dictionary
        
    Returns:
        Configured scheduler
    """
    return WarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=config.get('warmup_epochs', 5),
        total_epochs=config.get('num_epochs', 50),
        min_lr=0.0
    )
