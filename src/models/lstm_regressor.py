"""LSTM model architecture for regression."""

import torch
import torch.nn as nn
from typing import Tuple


class LSTMRegressor(nn.Module):
    """
    LSTM-based regression model for time series prediction.
    
    Features:
    - Multi-layer LSTM
    - Dropout for regularization
    - Fully connected output layer
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_size: int = 1
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout probability (applied between LSTM layers)
            output_size: Number of output targets
        """
        super(LSTMRegressor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            hidden: Optional initial hidden state tuple (h0, c0)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the output from the last time step
        # last_output shape: (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        # output shape: (batch_size, output_size)
        output = self.fc(last_output)
        
        return output
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state for LSTM.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Tuple of (h0, c0) hidden state tensors
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h0, c0)


def create_model(config: dict, input_size: int) -> LSTMRegressor:
    """
    Factory function to create LSTM model from configuration.
    
    Args:
        config: Configuration dictionary with model hyperparameters
        input_size: Number of input features
        
    Returns:
        Initialized LSTMRegressor model
    """
    return LSTMRegressor(
        input_size=input_size,
        hidden_size=config.get('hidden_size', 64),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3),
        output_size=1
    )
