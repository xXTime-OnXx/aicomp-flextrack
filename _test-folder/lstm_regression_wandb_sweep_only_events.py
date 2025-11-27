"""
PROPERLY DEBUGGED DR-ONLY TRAINING

Key fix: When training on DR-only, we must:
1. Get predictions ONLY for DR samples during training
2. Match metadata arrays to DR samples only
3. Still evaluate on ALL validation data
"""

import pandas as pd
import numpy as np
import os
import wandb
import datetime
import random
import holidays
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import sys
sys.path.append('C:/Github_FabianDubach/aicomp-flextrack/src/evaluation')

# Import both metric systems
try:
    from comp_metrics import evaluate_all_metrics
    HAS_COMP_METRICS = True
except:
    HAS_COMP_METRICS = False
    print("Warning: comp_metrics not found, using custom metrics only")

# Import custom DR-only metrics
sys.path.append('.')  # Add current directory
from custom_dr_metrics import evaluate_by_site_dr_only, print_metrics_summary

os.environ["WANDB_API_KEY"] = "3aaf9f796df65417b3f5f8560b43875171b55805"

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load data
df_train = pd.read_csv("C:/Github_FabianDubach/aicomp-flextrack/data/classification/classification-train.csv")
df_test = pd.read_csv("C:/Github_FabianDubach/aicomp-flextrack/data/classification/classification-test.csv")

print(f"Dataset shape: {df_train.shape}")
print(f"\nColumns: {df_train.columns.tolist()}")

australian_holidays = holidays.AU()

def add_time_features(df):
    df['Timestamp_Local'] = pd.to_datetime(df['Timestamp_Local'])
    df['hour'] = df['Timestamp_Local'].dt.hour
    df['minute'] = df['Timestamp_Local'].dt.minute
    df['month'] = df['Timestamp_Local'].dt.month
    df['day_of_week'] = df['Timestamp_Local'].dt.dayofweek
    df['is_weekend'] = df['Timestamp_Local'].dt.dayofweek >= 5
    df['is_holiday'] = df['Timestamp_Local'].dt.date.apply(lambda x: x in australian_holidays)
    return df

df_train = add_time_features(df_train)

df_train = df_train[
    ((df_train['hour'] > 6) & (df_train['hour'] < 19)) |
    ((df_train['hour'] == 6) & (df_train['minute'] >= 0)) |
    ((df_train['hour'] == 19) & (df_train['minute'] == 0))
].copy()

ENTRIES_PER_DAY = 53

df_train['month_sin'] = np.sin(2 * np.pi * (df_train['month'] / 12))
df_train['month_cos'] = np.cos(2 * np.pi * (df_train['month'] / 12))

df_train.drop(columns=['month'], inplace=True)

df_train['Building_Power_kW_diff_15min'] = (df_train.groupby('Site')['Building_Power_kW'].diff(1))
df_train['Building_Power_kW_diff_1h'] = (df_train.groupby('Site')['Building_Power_kW'].diff(4))
df_train['Building_Power_kW_diff_1d'] = (df_train.groupby('Site')['Building_Power_kW'].diff(ENTRIES_PER_DAY))

df_train['Dry_Bulb_Temperature_C_diff_15min'] = (df_train.groupby('Site')['Dry_Bulb_Temperature_C'].diff(1))
df_train['Global_Horizontal_Radiation_W/m2_diff_15min'] = (df_train.groupby('Site')['Global_Horizontal_Radiation_W/m2'].diff(1))

group = df_train.groupby('Site')['Building_Power_kW']

df_train['Building_Power_kW_rolling_mean_1h'] = group.rolling(4).mean().reset_index(level=0, drop=True)
df_train['Building_Power_kW_rolling_mean_2h'] = group.rolling(8).mean().reset_index(level=0, drop=True)
df_train['Building_Power_kW_rolling_mean_1d'] = group.rolling(ENTRIES_PER_DAY).mean().reset_index(level=0, drop=True)

df_train['Building_Power_kW_rolling_std_1h'] = group.rolling(4).std().reset_index(level=0, drop=True)
df_train['Building_Power_kW_rolling_std_2h'] = group.rolling(8).std().reset_index(level=0, drop=True)
df_train['Building_Power_kW_rolling_std_1d'] = group.rolling(ENTRIES_PER_DAY).std().reset_index(level=0, drop=True)

df_train['Building_Power_kW_rolling_min_1h'] = group.rolling(4).min().reset_index(level=0, drop=True)
df_train['Building_Power_kW_rolling_min_2h'] = group.rolling(8).min().reset_index(level=0, drop=True)

df_train['Building_Power_kW_rolling_max_1h'] = group.rolling(4).max().reset_index(level=0, drop=True)
df_train['Building_Power_kW_rolling_max_2h'] = group.rolling(8).max().reset_index(level=0, drop=True)

df_train = pd.get_dummies(df_train, columns=['minute', 'day_of_week'])
df_train = pd.get_dummies(df_train, columns=['Demand_Response_Flag'])

df_train['is_holiday'] = df_train['is_holiday'].astype(int)
df_train['is_weekend'] = df_train['is_weekend'].astype(int)

continuous_feature_columns = ['Dry_Bulb_Temperature_C', 'Global_Horizontal_Radiation_W/m2', 'Building_Power_kW',
                              'Building_Power_kW_diff_15min', 'Building_Power_kW_diff_1h', 'Building_Power_kW_diff_1d', 
                              'Dry_Bulb_Temperature_C_diff_15min', 'Global_Horizontal_Radiation_W/m2_diff_15min',
                              'Building_Power_kW_rolling_mean_1h', 'Building_Power_kW_rolling_mean_2h', 'Building_Power_kW_rolling_mean_1d',
                              'Building_Power_kW_rolling_std_1h', 'Building_Power_kW_rolling_std_2h', 'Building_Power_kW_rolling_std_1d',
                              'Building_Power_kW_rolling_min_1h', 'Building_Power_kW_rolling_min_2h', 'Building_Power_kW_rolling_max_1h', 
                              'Building_Power_kW_rolling_max_2h', 'hour']

categorical_feature_columns = ['minute_0', 'minute_15', 'minute_30', 'minute_45', 'day_of_week_0',
                               'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4',
                               'day_of_week_5', 'day_of_week_6', 'Demand_Response_Flag_-1',
                               'Demand_Response_Flag_0', 'Demand_Response_Flag_1', 
                               'is_weekend', 'is_holiday']

cyclic_feature_columns = ['month_sin', 'month_cos']

target_column = 'Demand_Response_Capacity_kW'

def prepare_data():
    """Prepare and return all data splits"""
    df_train_site_a = df_train[0:19345]
    df_train_site_b = df_train[19345:38690]
    df_train_site_c = df_train[38690:58035]

    X_continuous_site_a = df_train_site_a[continuous_feature_columns].values
    X_continuous_site_b = df_train_site_b[continuous_feature_columns].values
    X_continuous_site_c = df_train_site_c[continuous_feature_columns].values

    X_categorical_site_a = df_train_site_a[categorical_feature_columns].values
    X_categorical_site_b = df_train_site_b[categorical_feature_columns].values
    X_categorical_site_c = df_train_site_c[categorical_feature_columns].values

    X_cyclic_site_a = df_train_site_a[cyclic_feature_columns].values
    X_cyclic_site_b = df_train_site_b[cyclic_feature_columns].values
    X_cyclic_site_c = df_train_site_c[cyclic_feature_columns].values

    y_site_a = df_train_site_a[target_column].values.reshape(-1, 1)
    y_site_b = df_train_site_b[target_column].values.reshape(-1, 1)
    y_site_c = df_train_site_c[target_column].values.reshape(-1, 1)

    y_site_a_unscaled = y_site_a.copy()

    scaler_X_site_a = StandardScaler()
    scaler_X_site_b = StandardScaler()
    scaler_X_site_c = StandardScaler()

    scaler_y_site_a = StandardScaler()
    scaler_y_site_b = StandardScaler()
    scaler_y_site_c = StandardScaler()

    X_scaled_site_a = scaler_X_site_a.fit_transform(X_continuous_site_a)
    X_scaled_site_b = scaler_X_site_b.fit_transform(X_continuous_site_b)
    X_scaled_site_c = scaler_X_site_c.fit_transform(X_continuous_site_c)

    y_site_a = scaler_y_site_a.fit_transform(y_site_a)
    y_site_b = scaler_y_site_b.fit_transform(y_site_b)
    y_site_c = scaler_y_site_c.fit_transform(y_site_c)

    X_site_a = np.concatenate([X_scaled_site_a, X_categorical_site_a, X_cyclic_site_a], axis=1).astype(np.float32)
    X_site_b = np.concatenate([X_scaled_site_b, X_categorical_site_b, X_cyclic_site_b], axis=1).astype(np.float32)
    X_site_c = np.concatenate([X_scaled_site_c, X_categorical_site_c, X_cyclic_site_c], axis=1).astype(np.float32)

    y_site_a = y_site_a.astype(np.float32)
    y_site_b = y_site_b.astype(np.float32)
    y_site_c = y_site_c.astype(np.float32)

    # Remove NaN entries
    mask_site_a = np.ones(len(X_site_a), dtype=bool)
    mask_site_b = np.ones(len(X_site_b), dtype=bool)
    mask_site_c = np.ones(len(X_site_c), dtype=bool)

    mask_site_a[0:ENTRIES_PER_DAY] = False
    mask_site_b[0:ENTRIES_PER_DAY] = False
    mask_site_c[0:ENTRIES_PER_DAY] = False

    X_site_a = X_site_a[mask_site_a]
    X_site_b = X_site_b[mask_site_b]
    X_site_c = X_site_c[mask_site_c]

    y_site_a = y_site_a[mask_site_a]
    y_site_b = y_site_b[mask_site_b]
    y_site_c = y_site_c[mask_site_c]

    return {
        'X_site_a': X_site_a, 'X_site_b': X_site_b, 'X_site_c': X_site_c,
        'y_site_a': y_site_a, 'y_site_b': y_site_b, 'y_site_c': y_site_c,
        'scaler_X_site_a': scaler_X_site_a, 'scaler_X_site_b': scaler_X_site_b, 'scaler_X_site_c': scaler_X_site_c,
        'scaler_y_site_a': scaler_y_site_a, 'scaler_y_site_b': scaler_y_site_b, 'scaler_y_site_c': scaler_y_site_c,
        'y_site_a_unscaled': y_site_a_unscaled
    }


def create_sequences(X, y, seq_length):
    """Create sequences and return DR event mask for the target timestep"""
    sequences_X = []
    sequences_y = []
    dr_event_mask = []
    
    for i in range(len(X) - seq_length):
        sequences_X.append(X[i:i+seq_length])
        sequences_y.append(y[i+seq_length - 1])
        
        # Extract DR flag from the target timestep (last timestep in sequence)
        dr_flags_onehot = X[i+seq_length - 1, 30:33]
        dr_flag = np.argmax(dr_flags_onehot) - 1
        
        # Mark as True if this is an actual DR event (flag == -1 or flag == 1)
        is_dr_event = (dr_flag != 0)
        dr_event_mask.append(is_dr_event)

    print(f"Total sequences: {len(sequences_X)}")
    print(f"DR event sequences: {sum(dr_event_mask)} ({100*sum(dr_event_mask)/len(dr_event_mask):.1f}%)")
    
    return np.array(sequences_X), np.array(sequences_y), np.array(dr_event_mask)


class PowerDataset(Dataset):
    """Dataset that can optionally filter to only DR events"""
    def __init__(self, X, y, dr_event_mask=None, filter_dr_only=False):
        if filter_dr_only and dr_event_mask is not None:
            # Only keep samples where DR event occurred
            self.X = torch.FloatTensor(X[dr_event_mask])
            self.y = torch.FloatTensor(y[dr_event_mask])
            print(f"Dataset filtered to {len(self.X)} DR event samples (from {len(X)} total)")
        else:
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
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


def get_lr_with_warmup(epoch, base_lr, warmup_epochs):
    if warmup_epochs == 0 or epoch >= warmup_epochs:
        return base_lr
    else:
        return base_lr * (epoch + 1) / warmup_epochs


def train_model(config=None):
    """Training function with PROPER DEBUGGING"""
    wandb_run = wandb.init(
        project="AICOMP_Flextrack",
        entity="fabian-dubach-hochschule-luzern",
        config=config
    )
    config = wandb.config

    best_model_path = os.path.join(wandb_run.dir, "best_model.pt")
        
    print(f"\n{'='*60}")
    print(f"Starting DEBUGGED DR-only training")
    for key, value in dict(config).items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    # Prepare data
    data = prepare_data()
    
    # Create sequences WITH DR event masks
    print("\n--- Site A Sequences ---")
    X_train_seq_a, y_train_seq_a, dr_mask_a = create_sequences(
        data['X_site_a'], data['y_site_a'], config.sequence_length
    )
    
    print("\n--- Site C Sequences ---")
    X_train_seq_c, y_train_seq_c, dr_mask_c = create_sequences(
        data['X_site_c'], data['y_site_c'], config.sequence_length
    )
    
    print("\n--- Site B (Validation) Sequences ---")
    X_val_seq, y_val_seq, dr_mask_val = create_sequences(
        data['X_site_b'], data['y_site_b'], config.sequence_length
    )

    # Combine training data from sites A and C
    X_train_seq = np.vstack((X_train_seq_a, X_train_seq_c))
    y_train_seq = np.vstack((y_train_seq_a, y_train_seq_c))
    dr_mask_train = np.concatenate((dr_mask_a, dr_mask_c))

    print(f"\n--- Combined Training Data ---")
    print(f"Total training sequences: {len(X_train_seq)}")
    print(f"DR event sequences: {sum(dr_mask_train)} ({100*sum(dr_mask_train)/len(dr_mask_train):.1f}%)")

    # Create datasets - FILTER TO DR EVENTS ONLY FOR TRAINING
    train_dataset = PowerDataset(X_train_seq, y_train_seq, dr_mask_train, filter_dr_only=True)
    val_dataset = PowerDataset(X_val_seq, y_val_seq, dr_mask_val, filter_dr_only=False)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Extract metadata for DR-ONLY training samples (CRITICAL FIX!)
    # We need metadata that matches the FILTERED training data
    
    # Extract metadata from all sequences first
    train_building_power_scaled_a = X_train_seq_a[:, -1, 2]
    temp_a = np.zeros((len(train_building_power_scaled_a), 19))
    temp_a[:, 2] = train_building_power_scaled_a
    train_building_power_a_all = data['scaler_X_site_a'].inverse_transform(temp_a)[:, 2]
    train_demand_flags_a_all = np.argmax(X_train_seq_a[:, -1, 30:33], axis=1) - 1
    train_sites_a_all = np.array(['Site A'] * len(X_train_seq_a))
    
    train_building_power_scaled_c = X_train_seq_c[:, -1, 2]
    temp_c = np.zeros((len(train_building_power_scaled_c), 19))
    temp_c[:, 2] = train_building_power_scaled_c
    train_building_power_c_all = data['scaler_X_site_c'].inverse_transform(temp_c)[:, 2]
    train_demand_flags_c_all = np.argmax(X_train_seq_c[:, -1, 30:33], axis=1) - 1
    train_sites_c_all = np.array(['Site C'] * len(X_train_seq_c))
    
    # NOW FILTER METADATA TO DR EVENTS ONLY (matching filtered training data!)
    train_building_power_a_dr = train_building_power_a_all[dr_mask_a]
    train_demand_flags_a_dr = train_demand_flags_a_all[dr_mask_a]
    train_sites_a_dr = train_sites_a_all[dr_mask_a]
    
    train_building_power_c_dr = train_building_power_c_all[dr_mask_c]
    train_demand_flags_c_dr = train_demand_flags_c_all[dr_mask_c]
    train_sites_c_dr = train_sites_c_all[dr_mask_c]
    
    # Combine DR-only metadata
    train_building_power_dr = np.concatenate([train_building_power_a_dr, train_building_power_c_dr])
    train_demand_flags_dr = np.concatenate([train_demand_flags_a_dr, train_demand_flags_c_dr])
    train_sites_dr = np.concatenate([train_sites_a_dr, train_sites_c_dr])

    # For validation set (keep all)
    val_building_power_scaled = X_val_seq[:, -1, 2]
    temp_val = np.zeros((len(val_building_power_scaled), 19))
    temp_val[:, 2] = val_building_power_scaled
    val_building_power = data['scaler_X_site_b'].inverse_transform(temp_val)[:, 2]
    val_demand_flags = np.argmax(X_val_seq[:, -1, 30:33], axis=1) - 1
    val_sites = np.array(['Site B'] * len(X_val_seq))

    # CRITICAL: Verify lengths match FILTERED data
    print(f"\n--- CRITICAL: Data Integrity Check ---")
    print(f"Training DR dataset size: {len(train_dataset)}")
    print(f"Training DR building_power: {len(train_building_power_dr)}")
    print(f"Training DR demand_flags: {len(train_demand_flags_dr)}")
    print(f"Training DR sites: {len(train_sites_dr)}")
    assert len(train_dataset) == len(train_building_power_dr) == len(train_demand_flags_dr) == len(train_sites_dr), \
        f"MISMATCH! Dataset:{len(train_dataset)} vs metadata:{len(train_building_power_dr)}"
    print("✓ All training metadata arrays match FILTERED DR dataset")

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = SimpleRNN(
        input_size=X_train_seq.shape[2],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=1,
        dropout=config.dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Early stopping setup
    early_stopping_patience = 15
    best_nmae = float('inf')
    epochs_without_improvement = 0

    print("\nStarting training (on DR events only)...")
    
    for epoch in range(config.num_epochs):
        # Learning rate warmup
        current_lr = get_lr_with_warmup(epoch, config.learning_rate, config.warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Training - Standard MSE loss on DR events only
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
            clip_grad_norm_(model.parameters(), config.gradient_clip_val)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds_list.append(outputs.detach().cpu().numpy())
            train_targets_list.append(y_batch.cpu().numpy())

        train_loss /= len(train_loader)

        # Concatenate DR-only predictions and targets
        train_preds_dr = np.concatenate(train_preds_list)
        train_targets_dr = np.concatenate(train_targets_list)

        # Inverse transform DR-only predictions (MATCHING the filtered data!)
        # Split back to site A and C for proper inverse transform
        n_dr_a = sum(dr_mask_a)
        train_preds_dr_a = data['scaler_y_site_a'].inverse_transform(train_preds_dr[:n_dr_a])
        train_preds_dr_c = data['scaler_y_site_c'].inverse_transform(train_preds_dr[n_dr_a:])
        train_preds_original_dr = np.concatenate([train_preds_dr_a, train_preds_dr_c])

        train_targets_dr_a = data['scaler_y_site_a'].inverse_transform(train_targets_dr[:n_dr_a])
        train_targets_dr_c = data['scaler_y_site_c'].inverse_transform(train_targets_dr[n_dr_a:])
        train_targets_original_dr = np.concatenate([train_targets_dr_a, train_targets_dr_c])

        # DEBUG PRINTS (enable if issues)
        if epoch == 0:
            print(f"\n--- DEBUG INFO (Epoch 0) ---")
            print(f"Train preds shape: {train_preds_original_dr.shape}")
            print(f"Train targets shape: {train_targets_original_dr.shape}")
            print(f"Train building_power shape: {train_building_power_dr.shape}")
            print(f"Train demand_flags shape: {train_demand_flags_dr.shape}")
            print(f"Train sites shape: {train_sites_dr.shape}")
            print(f"Predictions range: [{train_preds_original_dr.min():.2f}, {train_preds_original_dr.max():.2f}]")
            print(f"Targets range: [{train_targets_original_dr.min():.2f}, {train_targets_original_dr.max():.2f}]")
            print(f"Manual MSE: {np.mean((train_preds_original_dr - train_targets_original_dr)**2):.4f}")
            print(f"----------------------------\n")

        # Calculate training metrics on DR-only data using CUSTOM METRICS
        train_metrics = evaluate_by_site_dr_only(
            y_true=train_targets_original_dr.flatten(),
            y_pred=train_preds_original_dr.flatten(),
            demand_flags=train_demand_flags_dr,
            site_labels=train_sites_dr,
            building_power=train_building_power_dr
        )

        # Validation on ALL validation data
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

        val_preds = np.concatenate(val_preds_list)
        val_targets = np.concatenate(val_targets_list)

        val_preds_original = data['scaler_y_site_b'].inverse_transform(val_preds)
        val_targets_original = data['scaler_y_site_b'].inverse_transform(val_targets)

        # Evaluate validation on DR events only using CUSTOM METRICS
        val_preds_dr = val_preds_original[dr_mask_val]
        val_targets_dr = val_targets_original[dr_mask_val]
        val_building_power_dr = val_building_power[dr_mask_val]
        val_demand_flags_dr = val_demand_flags[dr_mask_val]
        val_sites_dr = val_sites[dr_mask_val]

        val_metrics = evaluate_by_site_dr_only(
            y_true=val_targets_dr.flatten(),
            y_pred=val_preds_dr.flatten(),
            demand_flags=val_demand_flags_dr,
            site_labels=val_sites_dr,
            building_power=val_building_power_dr
        )

        # Log to W&B (using custom DR-only metrics)
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'val/loss': val_loss,
            'learning_rate': current_lr,
            
            # Training metrics (DR-only, custom metrics)
            'train/mae_kw': train_metrics['mae'],
            'train/rmse_kw': train_metrics['rmse'],
            'train/r2': train_metrics['r2'],
            'train/nmae_capacity': train_metrics['nmae_capacity_mean'],
            'train/nrmse_capacity': train_metrics['nrmse_capacity_mean'],
            'train/accuracy_direction': train_metrics['accuracy_direction'],
            'train/f1_direction': train_metrics['f1_direction'],
            'train/dr_samples': train_metrics['total_dr_samples'],
            
            # Validation metrics (DR-only, custom metrics)
            'val/mae_kw': val_metrics['mae'],
            'val/rmse_kw': val_metrics['rmse'],
            'val/r2': val_metrics['r2'],
            'val/nmae_capacity': val_metrics['nmae_capacity_mean'],
            'val/nrmse_capacity': val_metrics['nrmse_capacity_mean'],
            'val/accuracy_direction': val_metrics['accuracy_direction'],
            'val/f1_direction': val_metrics['f1_direction'],
            'val/dr_samples': val_metrics['total_dr_samples'],
        })

        # Console output
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'\nEpoch [{epoch+1}/{config.num_epochs}]')
            print(f'LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'Train: MAE={train_metrics["mae"]:.2f}kW, NMAE={train_metrics["nmae_capacity_mean"]:.2f}%, R²={train_metrics["r2"]:.3f}')
            print(f'Val:   MAE={val_metrics["mae"]:.2f}kW, NMAE={val_metrics["nmae_capacity_mean"]:.2f}%, R²={val_metrics["r2"]:.3f}')
        
        # Detailed summary every 20 epochs
        if (epoch + 1) % 20 == 0:
            print("\n" + "="*60)
            print("TRAINING METRICS (DR-only):")
            print(f"  MAE:  {train_metrics['mae']:.2f} kW")
            print(f"  RMSE: {train_metrics['rmse']:.2f} kW")
            print(f"  NMAE (capacity): {train_metrics['nmae_capacity_mean']:.2f}%")
            print(f"  R² Score: {train_metrics['r2']:.3f}")
            print(f"  Direction accuracy: {train_metrics['accuracy_direction']:.1f}%")
            print()
            print("VALIDATION METRICS (DR-only):")
            print(f"  MAE:  {val_metrics['mae']:.2f} kW")
            print(f"  RMSE: {val_metrics['rmse']:.2f} kW")
            print(f"  NMAE (capacity): {val_metrics['nmae_capacity_mean']:.2f}%")
            print(f"  R² Score: {val_metrics['r2']:.3f}")
            print(f"  Direction accuracy: {val_metrics['accuracy_direction']:.1f}%")
            print("="*60)

        # Early stopping based on validation NMAE (capacity-normalized)
        current_nmae = val_metrics['nmae_capacity_mean']
        if current_nmae < best_nmae:
            best_nmae = current_nmae
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ New best! Val NMAE: {best_nmae:.2f}%, Val MAE: {val_metrics['mae']:.2f}kW")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best Val NMAE: {best_nmae:.2f}%")
            break

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best Val NMAE (capacity): {best_nmae:.2f}%")
    print(f"{'='*60}\n")
    return best_nmae


# Main execution
if __name__ == "__main__":
    wandb.login()

    # SINGLE RUN MODE
    default_config = {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.4,
        "sequence_length": 12,
        "learning_rate": 0.0001,
        "weight_decay": 1e-3,
        "batch_size": 32,
        "num_epochs": 50,
        "gradient_clip_val": 1.0,
        "warmup_epochs": 5
    }

    train_model(default_config)