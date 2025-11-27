"""Custom PyTorch datasets for time series data."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series sequences.
    
    Creates sequences of specified length for LSTM training.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int,
        site_label: Optional[str] = None,
        building_power: Optional[np.ndarray] = None,
        demand_flags: Optional[np.ndarray] = None
    ):
        """
        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Target array of shape (n_samples, 1)
            sequence_length: Length of input sequences
            site_label: Optional site identifier for each sample
            building_power: Optional building power values for evaluation
            demand_flags: Optional demand response flags for evaluation
        """
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        self.site_label = site_label
        self.building_power = building_power
        self.demand_flags = demand_flags
        
    def __len__(self) -> int:
        """Return the number of valid sequences."""
        return len(self.X) - self.sequence_length + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its corresponding target.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (sequence, target) tensors
        """
        X_seq = self.X[idx:idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length - 1]
        
        return torch.FloatTensor(X_seq), torch.FloatTensor(y_target)
    
    def get_metadata(self, idx: int) -> dict:
        """
        Get metadata for a specific sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing metadata
        """
        actual_idx = idx + self.sequence_length - 1
        
        metadata = {'index': actual_idx}
        
        if self.site_label is not None:
            metadata['site'] = self.site_label
            
        if self.building_power is not None:
            metadata['building_power'] = self.building_power[actual_idx]
            
        if self.demand_flags is not None:
            metadata['demand_flag'] = self.demand_flags[actual_idx]
            
        return metadata


class MultiSiteDataset(Dataset):
    """
    Dataset that combines multiple sites for training.
    
    Useful for training on multiple sites simultaneously.
    """
    
    def __init__(self, datasets: list):
        """
        Args:
            datasets: List of TimeSeriesDataset objects
        """
        self.datasets = datasets
        self.lengths = [len(ds) for ds in datasets]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        
    def __len__(self) -> int:
        """Return total number of sequences across all sites."""
        return sum(self.lengths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence from the appropriate dataset.
        
        Args:
            idx: Global index across all datasets
            
        Returns:
            Tuple of (sequence, target) tensors
        """
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        
        return self.datasets[dataset_idx][local_idx]
