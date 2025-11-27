"""Data preparation and scaling utilities."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple
from utils.config import (
    CONTINUOUS_FEATURE_COLUMNS,
    CATEGORICAL_FEATURE_COLUMNS,
    CYCLIC_FEATURE_COLUMNS,
    TARGET_COLUMN,
    ENTRIES_PER_DAY
)


def prepare_site_data(
    df_train,
    site_ranges: Dict[str, Tuple[int, int]]
) -> Dict:
    """
    Prepare and scale data for all sites.
    
    Args:
        df_train: Preprocessed training dataframe
        site_ranges: Dictionary mapping site names to (start, end) indices
        
    Returns:
        Dictionary containing scaled data, scalers, and metadata for all sites
    """
    data = {}
    
    for site_name, (start_idx, end_idx) in site_ranges.items():
        df_site = df_train[start_idx:end_idx]
        
        # Extract features
        X_continuous = df_site[CONTINUOUS_FEATURE_COLUMNS].values
        X_categorical = df_site[CATEGORICAL_FEATURE_COLUMNS].values
        X_cyclic = df_site[CYCLIC_FEATURE_COLUMNS].values
        y = df_site[TARGET_COLUMN].values.reshape(-1, 1)
        
        # Store unscaled target for site_a (for evaluation)
        if site_name == 'site_a':
            data['y_site_a_unscaled'] = y.copy()
        
        # Create and fit scalers
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X_continuous)
        y_scaled = scaler_y.fit_transform(y)
        
        # Concatenate all features
        X_combined = np.concatenate([X_scaled, X_categorical, X_cyclic], axis=1).astype(np.float32)
        y_combined = y_scaled.astype(np.float32)
        
        # Remove NaN entries (from rolling/diff features)
        mask = np.ones(len(X_combined), dtype=bool)
        mask[0:ENTRIES_PER_DAY] = False
        
        X_combined = X_combined[mask]
        y_combined = y_combined[mask]
        
        # Store in dictionary
        data[f'X_{site_name}'] = X_combined
        data[f'y_{site_name}'] = y_combined
        data[f'scaler_X_{site_name}'] = scaler_X
        data[f'scaler_y_{site_name}'] = scaler_y
    
    return data


def prepare_data(df_train) -> Dict:
    """
    Prepare and return all data splits for the three sites.
    
    Args:
        df_train: Preprocessed training dataframe
        
    Returns:
        Dictionary containing all prepared data
    """
    site_ranges = {
        'site_a': (0, 19345),
        'site_b': (19345, 38690),
        'site_c': (38690, 58035)
    }
    
    return prepare_site_data(df_train, site_ranges)
