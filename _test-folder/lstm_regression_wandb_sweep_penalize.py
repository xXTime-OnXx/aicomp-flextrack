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

from comp_metrics import evaluate_all_metrics

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
    sequences_X = []
    sequences_y = []
    
    for i in range(len(X) - seq_length):
        sequences_X.append(X[i:i+seq_length])
        sequences_y.append(y[i+seq_length - 1])

    print(f"Sequences X: {len(sequences_X)}, Sequences Y: {len(sequences_y)}")
    
    return np.array(sequences_X), np.array(sequences_y)


class PowerDataset(Dataset):
    def __init__(self, X, y):
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
    """Training function that works with W&B sweep"""
    # Initialize W&B run (required for sweep)
    wandb_run = wandb.init(
        project="AICOMP_Flextrack",
        entity="fabian-dubach-hochschule-luzern",
        config=config
    )
    config = wandb.config

    # Path to save best model in current W&B run folder
    best_model_path = os.path.join(wandb_run.dir, "best_model.pt")
        
    print(f"\n{'='*60}")
    print(f"Starting run with config:")
    for key, value in dict(config).items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    # Prepare data
    data = prepare_data()
    
    # Create sequences
    X_train_seq_a, y_train_seq_a = create_sequences(data['X_site_a'], data['y_site_a'], config.sequence_length)
    X_train_seq_c, y_train_seq_c = create_sequences(data['X_site_c'], data['y_site_c'], config.sequence_length)
    X_val_seq, y_val_seq = create_sequences(data['X_site_b'], data['y_site_b'], config.sequence_length)

    X_train_seq = np.vstack((X_train_seq_a, X_train_seq_c))
    y_train_seq = np.vstack((y_train_seq_a, y_train_seq_c))

    # Create datasets and dataloaders
    train_dataset = PowerDataset(X_train_seq, y_train_seq)
    val_dataset = PowerDataset(X_val_seq, y_val_seq)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Extract building power and demand flags for metrics
    train_building_power_scaled = X_train_seq[:, -1, 2]
    temp = np.zeros((len(train_building_power_scaled), 19))
    temp[:, 2] = train_building_power_scaled
    temp_site_a = temp[:len(temp)//2]
    temp_site_c = temp[len(temp)//2:]
    train_building_power_site_a = data['scaler_X_site_a'].inverse_transform(temp_site_a)[:, 2]
    train_building_power_site_c = data['scaler_X_site_c'].inverse_transform(temp_site_c)[:, 2]
    train_building_power = np.concatenate([train_building_power_site_a, train_building_power_site_c])

    val_building_power_scaled = X_val_seq[:, -1, 2]
    temp_val = np.zeros((len(val_building_power_scaled), 19))
    temp_val[:, 2] = val_building_power_scaled
    val_building_power = data['scaler_X_site_b'].inverse_transform(temp_val)[:, 2]

    train_demand_flags = np.argmax(X_train_seq[:, -1, 30:33], axis=1) - 1
    val_demand_flags = np.argmax(X_val_seq[:, -1, 30:33], axis=1) - 1

    train_site_a = np.array(['Site A'] * len(train_building_power_site_a))
    train_site_c = np.array(['Site C'] * len(train_building_power_site_c))
    train_sites = np.concatenate([train_site_a, train_site_c])
    val_sites = np.array(['Site B'] * len(X_val_seq))

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    early_stopping_patience = 10
    best_nmae = float('inf')
    epochs_without_improvement = 0

    # Training loop
    print("Starting training...")
    
    for epoch in range(config.num_epochs):
        # Learning rate warmup
        current_lr = get_lr_with_warmup(epoch, config.learning_rate, config.warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Training
        model.train()
        train_loss = 0
        train_preds_list = []
        train_targets_list = []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # --- START: replaced loss calculation to penalize predictions during "no DR" events ---
            # Hyperparams (you can also put these in the sweep / config)
            penalty_weight = getattr(config, "fp_penalty_weight", 50.0)   # how strongly to punish false positives
            penalty_threshold = getattr(config, "fp_penalty_threshold", 0.1)  # threshold in scaled y-space

            outputs = model(X_batch)  # shape (B, 1)
            # base MSE (in scaled space)
            base_loss = nn.MSELoss()(outputs, y_batch)

            # Determine "no DR" mask from the one-hot demand flag slice.
            # Your code used X[..., 30:33] previously; keep same indices here.
            # argmax returns 0,1,2; earlier you used np.argmax(...) - 1 to map to [-1,0,1],
            # so argmax == 1 corresponds to Demand_Response_Flag == 0 (no event).
            dflag_onehots = X_batch[:, -1, 30:33]   # last timestep features, slice with one-hot DR flag
            dflag_ids = torch.argmax(dflag_onehots, dim=1)        # values in {0,1,2}
            mask_no_dr = (dflag_ids == 1)                        # boolean mask for "no event"

            # compute penalty only for no-DR samples
            if mask_no_dr.any():
                preds_no_dr = outputs[mask_no_dr].view(-1)  # (n_no_dr,)
                
                # penalize deviation from 0
                penalty_term = torch.mean(preds_no_dr ** 2)
            else:
                penalty_term = torch.tensor(0.0, device=outputs.device)


            loss = base_loss + penalty_weight * penalty_term

            # backward & optimize
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), config.gradient_clip_val)
            optimizer.step()
            # --- END: replaced loss calculation ---

            
            train_loss += loss.item()
            train_preds_list.append(outputs.detach().cpu().numpy())
            train_targets_list.append(y_batch.cpu().numpy())

        train_loss /= len(train_loader)

        # Concatenate predictions
        train_preds = np.concatenate(train_preds_list)
        train_targets = np.concatenate(train_targets_list)

        # Inverse transform
        train_preds_original_site_a = data['scaler_y_site_a'].inverse_transform(train_preds[:len(data['y_site_a'])])
        train_targets_original_site_a = data['scaler_y_site_a'].inverse_transform(train_targets[:len(data['y_site_a'])])
        train_preds_original_site_c = data['scaler_y_site_c'].inverse_transform(train_preds[len(data['y_site_a']):])
        train_targets_original_site_c = data['scaler_y_site_c'].inverse_transform(train_targets[len(data['y_site_a']):])

        train_preds_original = np.concatenate([train_preds_original_site_a, train_preds_original_site_c])
        train_targets_original = np.concatenate([train_targets_original_site_a, train_targets_original_site_c])

        # Calculate training metrics
        train_metrics = evaluate_all_metrics(
            y_true=train_targets_original,
            y_pred=train_preds_original,
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

        val_preds = np.concatenate(val_preds_list)
        val_targets = np.concatenate(val_targets_list)

        val_preds_original = data['scaler_y_site_b'].inverse_transform(val_preds)
        val_targets_original = data['scaler_y_site_b'].inverse_transform(val_targets)

        val_metrics = evaluate_all_metrics(
            y_true=val_targets_original.flatten(),
            y_pred=val_preds_original.flatten(),
            site_labels=val_sites,
            building_power=val_building_power,
            demand_flags=val_demand_flags
        )

        # Log to W&B
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'val/loss': val_loss,
            'learning_rate': current_lr,
            'train/nmae_range': train_metrics['nmae_range'],
            'train/nmae_mean': train_metrics['nmae_mean'],
            'train/nrmse_range': train_metrics['nrmse_range'],
            'train/nrmse_mean': train_metrics['nrmse_mean'],
            'train/geometric_mean_score': train_metrics['geometric_mean_score'],
            'train/f1_score': train_metrics['f1_score'],
            'val/nmae_range': val_metrics['nmae_range'],
            'val/nmae_mean': val_metrics['nmae_mean'],
            'val/nrmse_range': val_metrics['nrmse_range'],
            'val/nrmse_mean': val_metrics['nrmse_mean'],
            'val/geometric_mean_score': val_metrics['geometric_mean_score'],
            'val/f1_score': val_metrics['f1_score'],
            'train/base_loss': base_loss.item(),
            'train/penalty_term': penalty_term.item() if isinstance(penalty_term, torch.Tensor) else float(penalty_term),
            'train/total_loss': loss.item()
        })

        # Console output
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'\nEpoch [{epoch+1}/{config.num_epochs}]')
            print(f'Learning Rate: {current_lr:.6f}')
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'Val NMAE(mean): {val_metrics["nmae_mean"]:.2f}%')

        # Early stopping & save best model
        current_nmae = val_metrics['nmae_mean']
        if current_nmae < best_nmae:
            best_nmae = current_nmae
            epochs_without_improvement = 0
            
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model at epoch {epoch+1} with Val NMAE(mean): {best_nmae:.2f}%")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}. Best Val NMAE(mean): {best_nmae:.2f}%")
            break


    print(f"\nTraining completed! Best Val NMAE(mean): {best_nmae:.2f}%")
    return best_nmae


# Main execution
if __name__ == "__main__":
    wandb.login()

    # SWEEP MODE
    if "--sweep" in sys.argv:
        sweep_config = {
            "method": "bayes",
            "metric": {"name": "val/nmae_mean", "goal": "minimize"},
            "parameters": {
                # Wide, free, continuous LR search (SAFE)
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-6,
                    'max': 1e-2
                },

                # Wide batch sizes
                "batch_size": {"values": [32]},

                # Hidden size search from small → large LSTM
                "hidden_size": {"values": [32, 64, 128, 256]},

                # Depth search
                "num_layers": {"values": [1, 2, 3, 4]},

                # Fully continuous dropout
                "dropout": {
                    "distribution": "uniform",
                    "min": 0.0,
                    "max": 0.6
                },

                # Wide WD search over 3 orders of magnitude
                'weight_decay': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-6,
                    'max': 1e-2
                },

                # Free range of sequence lengths
                "sequence_length": {"values": [12, 24, 36, 48, 60]},

                "num_epochs": {"value": 50},
                "warmup_epochs": {"value": 5},
                'gradient_clip_val': {
                    'distribution': 'uniform',
                    'min': 0.5,
                    'max': 2.0
                },

                # inside sweep_config['parameters']:
                'fp_penalty_weight': {'values': [1.0, 10.0, 50.0, 100.0]},
                'fp_penalty_threshold': {'values': [0.01, 0.05, 0.1, 0.2]}
            }
        }

        sweep_id = wandb.sweep(
            sweep_config,
            project="AICOMP_Flextrack",
            entity="fabian-dubach-hochschule-luzern"
        )
        print(f"Sweep created: {sweep_id}")

        wandb.agent(
            sweep_id,
            function=train_model,
            count=500
        )

    # SINGLE RUN MODE
    else:
        default_config = {
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
            "fp_penalty_weight": 50,
            "fp_penalty_threshold": 0.1
        }

        train_model(default_config)