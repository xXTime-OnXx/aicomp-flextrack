#!/usr/bin/env python
"""Main training script for LSTM regression model."""

import os
import sys
import pandas as pd
import wandb

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from utils.config import set_seed, get_default_config
from transforms.preprocessing import preprocess_data
from utils.data_preparation import prepare_data
from trainers.lstm_trainer import LSTMTrainer
from evaluation.comp_metrics import evaluate_all_metrics

EVALUATION_FN = evaluate_all_metrics

# Set W&B API key (alternatively, use wandb.login() or set via environment)
os.environ["WANDB_API_KEY"] = "3aaf9f796df65417b3f5f8560b43875171b55805"

# Data paths (update these to your actual paths)
TRAIN_DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../data/regression/regression-train.csv'))
TEST_DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), '../data/regression/regression-test.csv'))


def train_model(config=None):
    """
    Train the LSTM model with given configuration.
    
    Args:
        config: Configuration dictionary. If None, uses default config.
        
    Returns:
        Best validation NMAE score
    """
    # Set random seed for reproducibility
    set_seed()
    
    # Use default config if none provided
    if config is None:
        config = get_default_config()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)
    
    print(f"Dataset shape: {df_train.shape}")
    print(f"Columns: {df_train.columns.tolist()}")
    
    # Apply transformations
    df_train = preprocess_data(df_train)
    
    # Prepare data for training
    print("Preparing data splits...")
    data = prepare_data(df_train)
    
    # Initialize trainer
    trainer = LSTMTrainer(
        config=config,
        data=data,
        wandb_project="AICOMP_Flextrack",
        wandb_entity="fabian-dubach-hochschule-luzern",
        evaluation_fn=EVALUATION_FN
    )
    
    # Train model
    best_nmae = trainer.train()
    
    return best_nmae


if __name__ == "__main__":
    # Login to W&B
    wandb.login()
    
    # Get default configuration
    config = get_default_config()
    
    # Train model
    train_model(config)
