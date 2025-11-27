"""Custom learning rate scheduler with warmup."""

import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupScheduler(_LRScheduler):
    """Learning rate scheduler with linear warmup."""
    
    def __init__(self, optimizer, warmup_epochs, last_epoch=-1):
        """
        Initialize the warmup scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_epochs: Number of warmup epochs
            last_epoch: The index of last epoch
        """
        self.warmup_epochs = warmup_epochs
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Constant learning rate after warmup
            return self.base_lrs
