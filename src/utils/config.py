"""Configuration utilities for the LSTM regression project."""

import random
import numpy as np
import torch


# Global constants
SEED = 42

# TODO: is this correct for every usage?
ENTRIES_PER_DAY = 96

# Feature column definitions
CONTINUOUS_FEATURE_COLUMNS = [
    'Dry_Bulb_Temperature_C', 'Global_Horizontal_Radiation_W/m2', 'Building_Power_kW',
    'Building_Power_kW_diff_15min', 'Building_Power_kW_diff_1h', 'Building_Power_kW_diff_1d',
    'Dry_Bulb_Temperature_C_diff_15min', 'Global_Horizontal_Radiation_W/m2_diff_15min',
    'Building_Power_kW_rolling_mean_1h', 'Building_Power_kW_rolling_mean_2h', 'Building_Power_kW_rolling_mean_1d',
    'Building_Power_kW_rolling_std_1h', 'Building_Power_kW_rolling_std_2h', 'Building_Power_kW_rolling_std_1d',
    'Building_Power_kW_rolling_min_1h', 'Building_Power_kW_rolling_min_2h', 'Building_Power_kW_rolling_max_1h',
    'Building_Power_kW_rolling_max_2h', 'hour'
]

CATEGORICAL_FEATURE_COLUMNS = [
    'minute_0', 'minute_15', 'minute_30', 'minute_45', 'day_of_week_0',
    'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4',
    'day_of_week_5', 'day_of_week_6', 'Demand_Response_Flag_-1',
    'Demand_Response_Flag_0', 'Demand_Response_Flag_1',
    'is_weekend', 'is_holiday'
]

CYCLIC_FEATURE_COLUMNS = ['month_sin', 'month_cos']

TARGET_COLUMN = 'Demand_Response_Capacity_kW'


def set_seed(seed: int = SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_default_config():
    """Get default training configuration."""
    return {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "sequence_length": 12,
        "learning_rate": 0.0001,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "num_epochs": 50,
        "gradient_clip_val": 1.0,
        "warmup_epochs": 5,
        "fp_penalty_weight": 10.0,
        "fp_penalty_threshold": 0.05,
    }


def get_sweep_config():
    """Get W&B sweep configuration."""
    return {
        "method": "bayes",
        "metric": {"name": "val/nmae_mean", "goal": "minimize"},
        "parameters": {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-2
            },
            "batch_size": {"values": [32]},
            "hidden_size": {"values": [32, 64, 128, 256]},
            "num_layers": {"values": [1, 2, 3, 4]},
            "dropout": {
                "distribution": "uniform",
                "min": 0.0,
                "max": 0.6
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-2
            },
            "sequence_length": {"values": [12, 24, 36, 48, 60]},
            "num_epochs": {"value": 50},
            "warmup_epochs": {"value": 5},
            'gradient_clip_val': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 2.0
            },
            'fp_penalty_weight': {'values': [1.0, 10.0, 50.0, 100.0]},
            'fp_penalty_threshold': {'values': [0.01, 0.05, 0.1, 0.2]}
        }
    }
