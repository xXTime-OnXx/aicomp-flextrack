"""
Complete RNN Regression Training with Wandb Sweep
This script includes data loading, feature engineering, model training, and hyperparameter sweeping.
"""

import pandas as pd
import numpy as np
import os
import wandb
from datetime import datetime
import random
import holidays
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
import sys

# Add path for comp_metrics (adjust this path as needed for your environment)
# sys.path.append('../src/evaluation')
# from comp_metrics import evaluate_all_metrics

# NOTE: Since we don't have access to comp_metrics, we'll create a simplified version
def evaluate_all_metrics(y_true, y_pred, site_labels, building_power, demand_flags):
    """Simplified metrics evaluation"""
    # Calculate range for normalization
    y_range = y_true.max() - y_true.min() if y_true.max() != y_true.min() else 1.0
    y_mean = y_true[y_true > 0].mean() if np.any(y_true > 0) else 1.0
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Normalized metrics
    nmae_range = (mae / y_range) * 100
    nmae_mean = (mae / y_mean) * 100
    nrmse_range = (rmse / y_range) * 100
    nrmse_mean = (rmse / y_mean) * 100
    
    # Geometric mean
    geometric_mean_score = np.sqrt(mae * rmse)
    
    # F1 score (binary: zero vs non-zero)
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    return {
        'nmae_range': nmae_range,
        'nmae_mean': nmae_mean,
        'nrmse_range': nrmse_range,
        'nrmse_mean': nrmse_mean,
        'geometric_mean_score': geometric_mean_score,
        'f1_score': f1
    }


# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Constants
ENTRIES_PER_DAY = 53  # 15 min intervals from 6:00 to 19:00

# Wandb API Key (update with your key or use environment variable)
os.environ["WANDB_API_KEY"] = "3aaf9f796df65417b3f5f8560b43875171b55805"
wandb.login()


def add_time_features(df):
    """Add temporal features to the dataframe"""
    australian_holidays = holidays.AU()
    
    df['Timestamp_Local'] = pd.to_datetime(df['Timestamp_Local'])
    df['hour'] = df['Timestamp_Local'].dt.hour
    df['minute'] = df['Timestamp_Local'].dt.minute
    df['month'] = df['Timestamp_Local'].dt.month
    df['day_of_week'] = df['Timestamp_Local'].dt.dayofweek
    df['is_weekend'] = df['Timestamp_Local'].dt.dayofweek >= 5
    df['is_holiday'] = df['Timestamp_Local'].dt.date.apply(lambda x: x in australian_holidays)
    return df


def engineer_features(df):
    """Complete feature engineering pipeline"""
    # Add time features
    df = add_time_features(df)
    
    # Remove night entries (6 AM to 7 PM only)
    df = df[
        ((df['hour'] > 6) & (df['hour'] < 19)) |
        ((df['hour'] == 6) & (df['minute'] >= 0)) |
        ((df['hour'] == 19) & (df['minute'] == 0))
    ].copy()
    
    # Cyclic encoding for month
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] / 12))
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] / 12))
    df.drop(columns=['month'], inplace=True)
    
    # Difference features
    df['Building_Power_kW_diff_15min'] = df.groupby('Site')['Building_Power_kW'].diff(1)
    df['Building_Power_kW_diff_1h'] = df.groupby('Site')['Building_Power_kW'].diff(4)
    df['Building_Power_kW_diff_1d'] = df.groupby('Site')['Building_Power_kW'].diff(ENTRIES_PER_DAY)
    df['Dry_Bulb_Temperature_C_diff_15min'] = df.groupby('Site')['Dry_Bulb_Temperature_C'].diff(1)
    df['Global_Horizontal_Radiation_W/m2_diff_15min'] = df.groupby('Site')['Global_Horizontal_Radiation_W/m2'].diff(1)
    
    # Rolling statistics
    group = df.groupby('Site')['Building_Power_kW']
    df['Building_Power_kW_rolling_mean_1h'] = group.rolling(4).mean().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_mean_2h'] = group.rolling(8).mean().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_mean_1d'] = group.rolling(ENTRIES_PER_DAY).mean().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_std_1h'] = group.rolling(4).std().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_std_2h'] = group.rolling(8).std().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_std_1d'] = group.rolling(ENTRIES_PER_DAY).std().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_min_1h'] = group.rolling(4).min().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_min_2h'] = group.rolling(8).min().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_max_1h'] = group.rolling(4).max().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_max_2h'] = group.rolling(8).max().reset_index(level=0, drop=True)
    
    # One-hot encodings
    df = pd.get_dummies(df, columns=['minute', 'day_of_week'])
    df = pd.get_dummies(df, columns=['Demand_Response_Flag'])
    df['is_holiday'] = df['is_holiday'].astype(int)
    df['is_weekend'] = df['is_weekend'].astype(int)
    
    return df


def prepare_data(df):
    """Prepare and normalize data"""
    continuous_feature_columns = [
        'Dry_Bulb_Temperature_C', 'Global_Horizontal_Radiation_W/m2', 'Building_Power_kW',
        'Building_Power_kW_diff_15min', 'Building_Power_kW_diff_1h', 'Building_Power_kW_diff_1d',
        'Dry_Bulb_Temperature_C_diff_15min', 'Global_Horizontal_Radiation_W/m2_diff_15min',
        'Building_Power_kW_rolling_mean_1h', 'Building_Power_kW_rolling_mean_2h', 'Building_Power_kW_rolling_mean_1d',
        'Building_Power_kW_rolling_std_1h', 'Building_Power_kW_rolling_std_2h', 'Building_Power_kW_rolling_std_1d',
        'Building_Power_kW_rolling_min_1h', 'Building_Power_kW_rolling_min_2h', 
        'Building_Power_kW_rolling_max_1h', 'Building_Power_kW_rolling_max_2h', 'hour'
    ]
    
    categorical_feature_columns = [
        'minute_0', 'minute_15', 'minute_30', 'minute_45',
        'day_of_week_0', 'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 
        'day_of_week_4', 'day_of_week_5', 'day_of_week_6',
        'Demand_Response_Flag_-1', 'Demand_Response_Flag_0', 'Demand_Response_Flag_1',
        'is_weekend', 'is_holiday'
    ]
    
    cyclic_feature_columns = ['month_sin', 'month_cos']
    target_column = 'Demand_Response_Capacity_kW'
    
    # Extract features
    X_continuous = df[continuous_feature_columns].values
    X_categorical = df[categorical_feature_columns].values
    X_cyclic = df[cyclic_feature_columns].values
    y = df[target_column].values.reshape(-1, 1)
    
    # Scale continuous features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X_continuous)
    y = scaler_y.fit_transform(y)
    
    # Concatenate all features
    X = np.concatenate([X_scaled, X_categorical, X_cyclic], axis=1)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Remove incomplete entries (first day of each site)
    mask = np.ones(len(X), dtype=bool)
    mask[0:ENTRIES_PER_DAY] = False  # Site A
    site_b_start = 365 * ENTRIES_PER_DAY
    mask[site_b_start:site_b_start + ENTRIES_PER_DAY] = False  # Site B
    site_c_start = 730 * ENTRIES_PER_DAY
    mask[site_c_start:site_c_start + ENTRIES_PER_DAY] = False  # Site C
    
    X = X[mask]
    y = y[mask]
    
    return X, y, scaler_X, scaler_y


def split_data(X, y):
    """Split data into train and validation sets"""
    site_a_entries = (365 - 1) * ENTRIES_PER_DAY
    site_b_entries = (365 - 1) * ENTRIES_PER_DAY
    train_end_idx = site_a_entries + site_b_entries
    
    X_train = X[:train_end_idx]
    X_val = X[train_end_idx:]
    y_train = y[:train_end_idx]
    y_val = y[train_end_idx:]
    
    return X_train, X_val, y_train, y_val


def create_sequences(X, y, sequence_length):
    """Create sequences for RNN"""
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    
    return np.array(X_seq), np.array(y_seq)


class PowerDataset(Dataset):
    """PyTorch Dataset for power data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleRNN(nn.Module):
    """LSTM model for regression"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


# Global variables for data (loaded once)
X_train_seq = None
X_val_seq = None
y_train_seq = None
y_val_seq = None
scaler_y = None
train_sites = None
val_sites = None
train_building_power = None
val_building_power = None
train_demand_flags = None
val_demand_flags = None


def load_and_prepare_data(data_path="../data/regression/"):
    """Load and prepare data (called once before sweep)"""
    global X_train_seq, X_val_seq, y_train_seq, y_val_seq, scaler_y
    global train_sites, val_sites, train_building_power, val_building_power
    global train_demand_flags, val_demand_flags
    
    print("Loading data...")
    df_train = pd.read_csv(f"{data_path}regression-train.csv")
    
    print("Engineering features...")
    df_train = engineer_features(df_train)
    
    print("Preparing data...")
    X, y, scaler_X, scaler_y = prepare_data(df_train)
    
    print("Splitting data...")
    X_train, X_val, y_train, y_val = split_data(X, y)
    
    print("Creating sequences...")
    sequence_length = 12  # 3 hours of data
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
    
    # Extract building power and demand flags for metrics
    train_building_power_scaled = X_train_seq[:, -1, 2]
    temp = np.zeros((len(train_building_power_scaled), 19))
    temp[:, 2] = train_building_power_scaled
    train_building_power = scaler_X.inverse_transform(temp)[:, 2]
    
    val_building_power_scaled = X_val_seq[:, -1, 2]
    temp_val = np.zeros((len(val_building_power_scaled), 19))
    temp_val[:, 2] = val_building_power_scaled
    val_building_power = scaler_X.inverse_transform(temp_val)[:, 2]
    
    train_demand_flags = np.argmax(X_train_seq[:, -1, 26:29], axis=1) - 1
    val_demand_flags = np.argmax(X_val_seq[:, -1, 26:29], axis=1) - 1
    
    train_sites = np.array(['Site A/B'] * len(X_train_seq))
    val_sites = np.array(['Site C'] * len(X_val_seq))
    
    print(f"Training samples: {len(X_train_seq)}")
    print(f"Validation samples: {len(X_val_seq)}")
    print(f"Input features: {X_train_seq.shape[2]}")


def objective_function_rnn(config=None):
    """Objective function for WandB sweep"""
    with wandb.init(
        entity="fabian-dubach-hochschule-luzern",
        project="AICOMP_Flextrack",
        name=f"rnn_sweep-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    ) as run:
        config = wandb.config
        
        # Create model
        model = SimpleRNN(
            input_size=X_train_seq.shape[2],
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=1,
            dropout=config.dropout
        ).to(device)
        
        # Create dataloaders
        train_dataset = PowerDataset(X_train_seq, y_train_seq)
        val_dataset = PowerDataset(X_val_seq, y_val_seq)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        best_val_loss = float('inf')
        best_val_nmae = float('inf')
        patience_counter = 0
        
        for epoch in range(config.epochs):
            # Training
            model.train()
            train_loss = 0
            train_preds_list = []
            train_targets_list = []
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                
                if config.gradient_clip_val > 0:
                    clip_grad_norm_(model.parameters(), config.gradient_clip_val)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_preds_list.append(outputs.detach().cpu().numpy())
                train_targets_list.append(y_batch.cpu().numpy())
            
            train_loss /= len(train_loader)
            
            # Concatenate predictions
            train_preds = np.concatenate(train_preds_list)
            train_targets = np.concatenate(train_targets_list)
            
            # Inverse transform
            train_preds_original = scaler_y.inverse_transform(train_preds)
            train_targets_original = scaler_y.inverse_transform(train_targets)
            
            # Calculate metrics
            train_metrics = evaluate_all_metrics(
                y_true=train_targets_original.flatten(),
                y_pred=train_preds_original.flatten(),
                site_labels=train_sites,
                building_power=train_building_power,
                demand_flags=train_demand_flags
            )
            
            # Validation
            model.eval()
            val_loss = 0
            val_preds_list = []
            val_targets_list = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    val_preds_list.append(outputs.cpu().numpy())
                    val_targets_list.append(y_batch.cpu().numpy())
            
            val_loss /= len(val_loader)
            
            # Concatenate predictions
            val_preds = np.concatenate(val_preds_list)
            val_targets = np.concatenate(val_targets_list)
            
            # Inverse transform
            val_preds_original = scaler_y.inverse_transform(val_preds)
            val_targets_original = scaler_y.inverse_transform(val_targets)
            
            # Calculate metrics
            val_metrics = evaluate_all_metrics(
                y_true=val_targets_original.flatten(),
                y_pred=val_preds_original.flatten(),
                site_labels=val_sites,
                building_power=val_building_power,
                demand_flags=val_demand_flags
            )
            
            # WandB logging
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                
                # Training metrics
                'train/nmae_range': train_metrics['nmae_range'],
                'train/nmae_mean': train_metrics['nmae_mean'],
                'train/nrmse_range': train_metrics['nrmse_range'],
                'train/nrmse_mean': train_metrics['nrmse_mean'],
                'train/geometric_mean_score': train_metrics['geometric_mean_score'],
                'train/f1_score': train_metrics['f1_score'],
                
                # Validation metrics
                'val/nmae_range': val_metrics['nmae_range'],
                'val/nmae_mean': val_metrics['nmae_mean'],
                'val/nrmse_range': val_metrics['nrmse_range'],
                'val/nrmse_mean': val_metrics['nrmse_mean'],
                'val/geometric_mean_score': val_metrics['geometric_mean_score'],
                'val/f1_score': val_metrics['f1_score'],
            })
            
            # Early stopping based on NMAE
            if val_metrics['nmae_mean'] < best_val_nmae:
                best_val_nmae = val_metrics['nmae_mean']
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Log best metrics
        wandb.log({
            "best_val_loss": best_val_loss,
            "best_val_nmae_mean": best_val_nmae
        })
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        return best_val_nmae


def run_sweep_rnn(count=50):
    """Run WandB sweep for RNN hyperparameter optimization"""
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val/nmae_mean', 'goal': 'minimize'},
        'name': 'rnn_regression_sweep_nmae',
        'parameters': {
            'batch_size': {
                'values': [16, 32, 64]
            },
            'hidden_size': {
                'values': [32, 64, 128, 256]
            },
            'num_layers': {
                'values': [1, 2, 3]
            },
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-2
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-3
            },
            'dropout': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.5
            },
            'gradient_clip_val': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 5.0
            },
            'epochs': {
                'values': [50]
            },
            'patience': {
                'values': [10]
            }
        }
    }
    
    sweep_id = wandb.sweep(
        sweep_config, 
        project="AICOMP_Flextrack", 
        entity="fabian-dubach-hochschule-luzern"
    )
    
    wandb.agent(sweep_id, function=objective_function_rnn, count=count)


if __name__ == "__main__":
    # Load and prepare data once
    load_and_prepare_data(data_path="../data/regression/")
    
    # Run sweep
    print("Starting RNN regression sweep...")
    run_sweep_rnn(count=50)