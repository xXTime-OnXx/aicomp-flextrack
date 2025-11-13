"""
LSTM Regression Sweep Script with Bayesian Optimization and Learning Rate Warmup
Uses the exact same feature engineering and scaling as the original notebook
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
import sys

# Add path to custom evaluation metrics if needed
sys.path.append('../src/evaluation')

# Conditional import for custom metrics
try:
    from comp_metrics import evaluate_all_metrics
    HAS_CUSTOM_METRICS = True
except ImportError:
    print("Warning: comp_metrics not found. Using fallback metrics.")
    HAS_CUSTOM_METRICS = False

# ============== CONFIGURATION ==============
os.environ["WANDB_API_KEY"] = "3aaf9f796df65417b3f5f8560b43875171b55805"

SEED = 42
ENTRIES_PER_DAY = 53  # 15 min intervals from 6:00 to 19:00

# Set seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============== FEATURE ENGINEERING FUNCTIONS ==============
australian_holidays = holidays.AU()

def add_time_features(df):
    """Add temporal features exactly as in the notebook"""
    df['Timestamp_Local'] = pd.to_datetime(df['Timestamp_Local'])
    df['hour'] = df['Timestamp_Local'].dt.hour
    df['minute'] = df['Timestamp_Local'].dt.minute
    df['month'] = df['Timestamp_Local'].dt.month
    df['day_of_week'] = df['Timestamp_Local'].dt.dayofweek
    df['is_weekend'] = df['Timestamp_Local'].dt.dayofweek >= 5
    df['is_holiday'] = df['Timestamp_Local'].dt.date.apply(lambda x: x in australian_holidays)
    return df

def remove_night_data(df):
    """Remove entries over night (6 AM to 7 PM only)"""
    df = df[
        ((df['hour'] > 6) & (df['hour'] < 19)) |
        ((df['hour'] == 6) & (df['minute'] >= 0)) |
        ((df['hour'] == 19) & (df['minute'] == 0))
    ].copy()
    return df

def add_cyclic_encoding(df):
    """Add cyclic encoding for month"""
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] / 12))
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] / 12))
    df.drop(columns=['month'], inplace=True)
    return df

def add_difference_features(df):
    """Add difference features"""
    df['Building_Power_kW_diff_15min'] = df.groupby('Site')['Building_Power_kW'].diff(1)
    df['Building_Power_kW_diff_1h'] = df.groupby('Site')['Building_Power_kW'].diff(4)
    df['Building_Power_kW_diff_1d'] = df.groupby('Site')['Building_Power_kW'].diff(ENTRIES_PER_DAY)
    
    df['Dry_Bulb_Temperature_C_diff_15min'] = df.groupby('Site')['Dry_Bulb_Temperature_C'].diff(1)
    df['Global_Horizontal_Radiation_W/m2_diff_15min'] = df.groupby('Site')['Global_Horizontal_Radiation_W/m2'].diff(1)
    return df

def add_rolling_statistics(df):
    """Add rolling statistics"""
    group = df.groupby('Site')['Building_Power_kW']
    
    # Rolling means
    df['Building_Power_kW_rolling_mean_1h'] = group.rolling(4).mean().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_mean_2h'] = group.rolling(8).mean().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_mean_1d'] = group.rolling(ENTRIES_PER_DAY).mean().reset_index(level=0, drop=True)
    
    # Rolling std
    df['Building_Power_kW_rolling_std_1h'] = group.rolling(4).std().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_std_2h'] = group.rolling(8).std().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_std_1d'] = group.rolling(ENTRIES_PER_DAY).std().reset_index(level=0, drop=True)
    
    # Rolling min/max
    df['Building_Power_kW_rolling_min_1h'] = group.rolling(4).min().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_min_2h'] = group.rolling(8).min().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_max_1h'] = group.rolling(4).max().reset_index(level=0, drop=True)
    df['Building_Power_kW_rolling_max_2h'] = group.rolling(8).max().reset_index(level=0, drop=True)
    
    return df

def engineer_features(df):
    """Complete feature engineering pipeline"""
    df = add_time_features(df)
    df = remove_night_data(df)
    df = add_cyclic_encoding(df)
    df = add_difference_features(df)
    df = add_rolling_statistics(df)
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['minute', 'day_of_week'], prefix=['minute', 'day_of_week'])
    df = pd.get_dummies(df, columns=['Demand_Response_Flag'], prefix='Demand_Response_Flag')
    
    # Convert boolean columns to int
    df['is_holiday'] = df['is_holiday'].astype(int)
    df['is_weekend'] = df['is_weekend'].astype(int)
    
    return df

# ============== LEARNING RATE WARMUP ==============
def get_lr_with_warmup(epoch, base_lr, warmup_epochs):
    """
    Calculate learning rate with linear warmup
    
    Args:
        epoch: Current epoch (0-indexed)
        base_lr: Target learning rate after warmup
        warmup_epochs: Number of epochs for warmup
    
    Returns:
        Current learning rate
    """
    if warmup_epochs == 0 or epoch >= warmup_epochs:
        return base_lr
    else:
        # Linear warmup from 0 to base_lr
        return base_lr * (epoch + 1) / warmup_epochs

# ============== DATASET CLASS ==============
class PowerDataset(Dataset):
    """PyTorch Dataset for power consumption data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============== MODEL ARCHITECTURE ==============
class SimpleRNN(nn.Module):
    """LSTM/GRU model for time series regression"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3, model_type='LSTM'):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        
        # Choose RNN type
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        # Take the last time step
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ============== SEQUENCE CREATION ==============
def create_sequences(X, y, seq_length):
    """Create sequences for RNN training"""
    sequences_X = []
    sequences_y = []
    
    for i in range(len(X) - seq_length):
        sequences_X.append(X[i:i+seq_length])
        sequences_y.append(y[i+seq_length])
    
    return np.array(sequences_X), np.array(sequences_y)

# ============== FALLBACK METRICS ==============
def calculate_fallback_metrics(y_true, y_pred):
    """Fallback metrics when custom metrics are not available"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Normalized metrics
    y_range = y_true.max() - y_true.min()
    y_mean = y_true.mean()
    
    nmae_range = (mae / y_range * 100) if y_range > 0 else 0
    nmae_mean = (mae / y_mean * 100) if y_mean > 0 else 0
    nrmse_range = (rmse / y_range * 100) if y_range > 0 else 0
    nrmse_mean = (rmse / y_mean * 100) if y_mean > 0 else 0
    
    # Geometric mean
    geometric_mean = np.sqrt(mae * rmse)
    
    # F1 score (binary: zero vs non-zero)
    y_true_binary = (y_true > 0).astype(int).flatten()
    y_pred_binary = (y_pred > 0).astype(int).flatten()
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    return {
        'nmae_range': nmae_range,
        'nmae_mean': nmae_mean,
        'nrmse_range': nrmse_range,
        'nrmse_mean': nrmse_mean,
        'geometric_mean_score': geometric_mean,
        'f1_score': f1
    }

# ============== DATA PREPARATION ==============
def prepare_data(config):
    """Load and prepare data with feature engineering"""
    # Load data
    df_train = pd.read_csv("C:/Github_FabianDubach/aicomp-flextrack/data/regression/regression-train.csv")
    
    # Feature engineering
    df_train = engineer_features(df_train)
    
    # Define feature columns (EXACTLY as in notebook)
    continuous_feature_columns = [
        'Dry_Bulb_Temperature_C', 'Global_Horizontal_Radiation_W/m2', 'Building_Power_kW',
        'Building_Power_kW_diff_15min', 'Building_Power_kW_diff_1h', 'Building_Power_kW_diff_1d',
        'Dry_Bulb_Temperature_C_diff_15min', 'Global_Horizontal_Radiation_W/m2_diff_15min',
        'Building_Power_kW_rolling_mean_1h', 'Building_Power_kW_rolling_mean_2h', 'Building_Power_kW_rolling_mean_1d',
        'Building_Power_kW_rolling_std_1h', 'Building_Power_kW_rolling_std_2h', 'Building_Power_kW_rolling_std_1d',
        'Building_Power_kW_rolling_min_1h', 'Building_Power_kW_rolling_min_2h', 'Building_Power_kW_rolling_max_1h',
        'Building_Power_kW_rolling_max_2h', 'hour'
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
    
    # Split by site (same as notebook)
    df_train_site_a = df_train[0:19345]
    df_train_site_b = df_train[19345:38690]
    df_train_site_c = df_train[38690:58035]
    
    # Extract features and targets
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
    
    # Scaling (site-specific scalers)
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
    
    # Concatenate features
    X_site_a = np.concatenate([X_scaled_site_a, X_categorical_site_a, X_cyclic_site_a], axis=1)
    X_site_b = np.concatenate([X_scaled_site_b, X_categorical_site_b, X_cyclic_site_b], axis=1)
    X_site_c = np.concatenate([X_scaled_site_c, X_categorical_site_c, X_cyclic_site_c], axis=1)
    
    # Convert to float32
    X_site_a = X_site_a.astype(np.float32)
    X_site_b = X_site_b.astype(np.float32)
    X_site_c = X_site_c.astype(np.float32)
    
    y_site_a = y_site_a.astype(np.float32)
    y_site_b = y_site_b.astype(np.float32)
    y_site_c = y_site_c.astype(np.float32)
    
    # Remove incomplete entries (first day)
    X_site_a = X_site_a[ENTRIES_PER_DAY:]
    X_site_b = X_site_b[ENTRIES_PER_DAY:]
    X_site_c = X_site_c[ENTRIES_PER_DAY:]
    
    y_site_a = y_site_a[ENTRIES_PER_DAY:]
    y_site_b = y_site_b[ENTRIES_PER_DAY:]
    y_site_c = y_site_c[ENTRIES_PER_DAY:]
    
    # Create sequences
    seq_length = config['sequence_length']
    X_train_seq_a, y_train_seq_a = create_sequences(X_site_a, y_site_a, seq_length)
    X_train_seq_c, y_train_seq_c = create_sequences(X_site_c, y_site_c, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_site_b, y_site_b, seq_length)
    
    # Combine training data (Site A + Site C)
    X_train_seq = np.vstack((X_train_seq_a, X_train_seq_c))
    y_train_seq = np.vstack((y_train_seq_a, y_train_seq_c))
    
    # Extract metadata for metrics
    # Building_Power_kW is at index 2 in continuous features (as per notebook)
    building_power_idx = 2
    
    train_building_power_scaled = X_train_seq[:, -1, building_power_idx]
    temp = np.zeros((len(train_building_power_scaled), len(continuous_feature_columns)))
    temp[:, building_power_idx] = train_building_power_scaled
    temp_site_a = temp[:len(temp)//2]   # first half = Site A
    temp_site_c = temp[len(temp)//2:]   # second half = Site C
    train_building_power_site_a = scaler_X_site_a.inverse_transform(temp_site_a)[:, building_power_idx]
    train_building_power_site_c = scaler_X_site_c.inverse_transform(temp_site_c)[:, building_power_idx]
    train_building_power = np.concatenate([train_building_power_site_a, train_building_power_site_c])
    
    val_building_power_scaled = X_val_seq[:, -1, building_power_idx]
    temp_val = np.zeros((len(val_building_power_scaled), len(continuous_feature_columns)))
    temp_val[:, building_power_idx] = val_building_power_scaled
    val_building_power = scaler_X_site_b.inverse_transform(temp_val)[:, building_power_idx]
    
    # Demand response flags at indices 30, 31, 32 for Demand_Response_Flag_-1, 0, 1
    train_demand_flags = np.argmax(X_train_seq[:, -1, 30:33], axis=1) - 1
    val_demand_flags = np.argmax(X_val_seq[:, -1, 30:33], axis=1) - 1
    
    # Site labels
    train_site_a_labels = np.array(['Site A'] * len(train_building_power_site_a))
    train_site_c_labels = np.array(['Site C'] * len(train_building_power_site_c))
    train_sites = np.concatenate([train_site_a_labels, train_site_c_labels])
    val_sites = np.array(['Site B'] * len(X_val_seq))
    
    # Return all necessary data
    return {
        'X_train': X_train_seq,
        'y_train': y_train_seq,
        'X_val': X_val_seq,
        'y_val': y_val_seq,
        'scalers': {
            'scaler_y_site_a': scaler_y_site_a,
            'scaler_y_site_b': scaler_y_site_b,
            'scaler_y_site_c': scaler_y_site_c,
        },
        'metadata': {
            'train_building_power': train_building_power,
            'val_building_power': val_building_power,
            'train_demand_flags': train_demand_flags,
            'val_demand_flags': val_demand_flags,
            'train_sites': train_sites,
            'val_sites': val_sites,
            'y_site_a': len(X_train_seq_a)  # Store length of site A sequences for splitting
        },
        'input_size': X_train_seq.shape[2]
    }

# ============== TRAINING FUNCTION ==============
def train_model(config=None):
    """Training function for Wandb sweep with learning rate warmup"""
    # Initialize wandb
    with wandb.init(config=config):
        config = wandb.config
        
        print("Preparing data...")
        data = prepare_data(config)
        
        # Datasets and loaders
        train_dataset = PowerDataset(data['X_train'], data['y_train'])
        val_dataset = PowerDataset(data['X_val'], data['y_val'])
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Model setup
        model = SimpleRNN(
            input_size=data['input_size'],
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=1,
            dropout=config.dropout,
            model_type=config.model_type
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Tracking
        best_val_nmae = float('inf')
        epochs_no_improve = 0
        early_stop_patience = 5
        
        print(f"Warmup enabled: {config.warmup_epochs} epochs")
        
        for epoch in range(config.num_epochs):
            # ============== LEARNING RATE WARMUP ==============
            current_lr = get_lr_with_warmup(
                epoch, 
                config.learning_rate, 
                config.warmup_epochs
            )
            
            # Update optimizer learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # === TRAIN ===
            model.train()
            train_loss = 0
            train_preds_list, train_targets_list = [], []
            
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
            train_preds = np.concatenate(train_preds_list)
            train_targets = np.concatenate(train_targets_list)
            
            # Inverse transform - split by original y_site_a length
            y_site_a_len = data['metadata']['y_site_a']  # Already an int, don't call len()
            train_preds_site_a = data['scalers']['scaler_y_site_a'].inverse_transform(train_preds[:y_site_a_len])
            train_targets_site_a = data['scalers']['scaler_y_site_a'].inverse_transform(train_targets[:y_site_a_len])
            train_preds_site_c = data['scalers']['scaler_y_site_c'].inverse_transform(train_preds[y_site_a_len:])
            train_targets_site_c = data['scalers']['scaler_y_site_c'].inverse_transform(train_targets[y_site_a_len:])
            
            train_preds_original = np.concatenate([train_preds_site_a, train_preds_site_c])
            train_targets_original = np.concatenate([train_targets_site_a, train_targets_site_c])
            
            # Metrics
            if HAS_CUSTOM_METRICS:
                train_metrics = evaluate_all_metrics(
                    y_true=train_targets_original,  # NOT flattened
                    y_pred=train_preds_original,    # NOT flattened
                    site_labels=data['metadata']['train_sites'],
                    building_power=data['metadata']['train_building_power'],
                    demand_flags=data['metadata']['train_demand_flags']
                )
            else:
                train_metrics = calculate_fallback_metrics(train_targets_original, train_preds_original)
            
            # === VALIDATION ===
            model.eval()
            val_loss = 0
            val_preds_list, val_targets_list = [], []
            
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
            
            # Inverse transform
            val_preds_original = data['scalers']['scaler_y_site_b'].inverse_transform(val_preds)
            val_targets_original = data['scalers']['scaler_y_site_b'].inverse_transform(val_targets)
            
            if HAS_CUSTOM_METRICS:
                val_metrics = evaluate_all_metrics(
                    y_true=val_targets_original.flatten(),
                    y_pred=val_preds_original.flatten(),
                    site_labels=data['metadata']['val_sites'],
                    building_power=data['metadata']['val_building_power'],
                    demand_flags=data['metadata']['val_demand_flags']
                )
            else:
                val_metrics = calculate_fallback_metrics(val_targets_original, val_preds_original)
            
            # === LOGGING ===
            # Sanity check for predictions
            val_pred_mean = val_preds_original.mean()
            val_pred_std = val_preds_original.std()
            val_pred_nonzero_pct = (np.abs(val_preds_original) > 1e-6).mean() * 100
            
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'learning_rate': current_lr,  # Log actual learning rate being used
                'train/nmae_mean': train_metrics['nmae_mean'],
                'val/nmae_mean': val_metrics['nmae_mean'],
                'train/nrmse_mean': train_metrics['nrmse_mean'],
                'val/nrmse_mean': val_metrics['nrmse_mean'],
                'val/pred_mean': val_pred_mean,
                'val/pred_std': val_pred_std,
                'val/pred_nonzero_pct': val_pred_nonzero_pct,
            })
            
            # Warning if model is predicting near-zero values
            if val_pred_nonzero_pct < 10:
                print(f"⚠️  WARNING: Model predicting mostly zeros ({val_pred_nonzero_pct:.1f}% non-zero)")
            
            # === SAVE CHECKPOINT EACH EPOCH ===
            checkpoint_path = os.path.join(wandb.run.dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            
            # === SAVE BEST MODEL BASED ON LOWEST NMAE ===
            current_val_nmae = val_metrics['nmae_mean']
            
            if current_val_nmae < best_val_nmae:
                best_val_nmae = current_val_nmae
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pt'))
                epochs_no_improve = 0
                print(f"✅ Epoch {epoch+1}: New best model saved with NMAE {best_val_nmae:.4f} (LR: {current_lr:.6f})")
            else:
                epochs_no_improve += 1
                print(f"⚠️  Epoch {epoch+1}: No improvement ({epochs_no_improve}/{early_stop_patience}) (LR: {current_lr:.6f})")
            
            # === EARLY STOPPING ===
            if epochs_no_improve >= early_stop_patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1} — no improvement in {early_stop_patience} epochs.")
                break

        print("Training completed!")


# ============== SWEEP CONFIGURATION ==============
sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'val/nmae_mean',
        'goal': 'minimize'
    },
    'parameters': {
        # Model architecture
        'hidden_size': {
            'values': [32, 64, 128, 256]
        },
        'num_layers': {
            'values': [1, 2, 3, 4]
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        },
        'sequence_length': {
            'values': [6, 12, 24, 36]  # Different lookback windows
        },
        'model_type': {
            'values': ['LSTM', 'GRU']
        },
        
        # Training hyperparameters
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
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'gradient_clip_val': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 2.0
        },
        
        # Learning rate warmup (NEW)
        'warmup_epochs': {
            'values': [0, 3, 5, 7, 10]  # Different warmup durations (0 = disabled)
        },
        
        # Fixed parameters
        'num_epochs': {
            'value': 50
        },
        'optimizer': {
            'value': 'Adam'
        },
        'loss_function': {
            'value': 'MSE'
        }
    }
}

# ============== MAIN EXECUTION ==============
if __name__ == "__main__":
    # Login to wandb
    wandb.login()
    
    # Initialize sweep
    sweep_id = wandb.sweep(
        sweep_config,
        entity="fabian-dubach-hochschule-luzern",
        project="AICOMP_Flextrack"
    )
    
    # Run sweep
    print(f"Starting sweep with ID: {sweep_id}")
    print("Sweep URL: https://wandb.ai/fabian-dubach-hochschule-luzern/AICOMP_Flextrack/sweeps/" + sweep_id)
    
    # Run the sweep agent (can specify count for number of runs)
    wandb.agent(sweep_id, train_model, count=50)  # Adjust count as needed
    
    print("Sweep completed!")