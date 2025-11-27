"""Script to download and prepare the dataset."""

import os
import pandas as pd
from pathlib import Path


def download_data(data_dir='../data'):
    """
    Download or load the dataset.
    
    Args:
        data_dir: Directory to store the data
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading dataset...")
    
    # Update these paths to your actual data locations
    train_path = data_path / "classification-train.csv"
    test_path = data_path / "classification-test.csv"
    
    if not train_path.exists() or not test_path.exists():
        print("Please place your data files in the data directory:")
        print(f"  - {train_path}")
        print(f"  - {test_path}")
        return None, None
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"Training data shape: {df_train.shape}")
    print(f"Test data shape: {df_test.shape}")
    print(f"\nColumns: {df_train.columns.tolist()}")
    
    return df_train, df_test


if __name__ == "__main__":
    download_data()
