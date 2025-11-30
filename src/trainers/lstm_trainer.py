"""Model trainer with W&B integration and custom loss."""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import wandb
from typing import Dict, Optional

from datasets.timeseries_dataset import TimeSeriesDataset, MultiSiteDataset
from models.lstm_regressor import create_model
from schedulers.warmup_cosine import create_scheduler


class LSTMTrainer:
    """
    Trainer class for LSTM model with W&B logging and custom penalties.
    
    Features:
    - W&B integration for experiment tracking
    - Custom loss with false positive penalty
    - Early stopping
    - Learning rate warmup
    - Gradient clipping
    """
    
    def __init__(
        self,
        config: dict,
        data: Dict,
        wandb_project: str = "AICOMP_Flextrack",
        wandb_entity: str = "fabian-dubach-hochschule-luzern",
        evaluation_fn: Optional[callable] = None
    ):
        """
        Args:
            config: Configuration dictionary with hyperparameters
            data: Prepared data dictionary from data_preparation module
            wandb_project: W&B project name
            wandb_entity: W&B entity name
            evaluation_fn: Optional evaluation function for metrics
        """
        self.config = config
        self.data = data
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.evaluation_fn = evaluation_fn
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize W&B
        self.wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config=config
        )
        self.config = wandb.config
        
        # Model save path
        self.best_model_path = os.path.join(self.wandb_run.dir, "best_model.pt")
        
    def create_sequences(self, X: np.ndarray, y: np.ndarray, seq_length: int):
        """
        Create sequences from time series data.
        
        Args:
            X: Feature array
            y: Target array
            seq_length: Sequence length
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        sequences_X = []
        sequences_y = []
        
        for i in range(len(X) - seq_length):
            sequences_X.append(X[i:i+seq_length])
            sequences_y.append(y[i+seq_length])
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def prepare_dataloaders(self):
        """Prepare training and validation dataloaders."""
        seq_len = self.config.sequence_length
        
        # Create sequences
        X_train_seq_a, y_train_seq_a = self.create_sequences(
            self.data['X_site_a'], self.data['y_site_a'], seq_len
        )
        X_train_seq_c, y_train_seq_c = self.create_sequences(
            self.data['X_site_c'], self.data['y_site_c'], seq_len
        )
        X_val_seq, y_val_seq = self.create_sequences(
            self.data['X_site_b'], self.data['y_site_b'], seq_len
        )
        
        # Combine training sites
        X_train_seq = np.vstack((X_train_seq_a, X_train_seq_c))
        y_train_seq = np.vstack((y_train_seq_a, y_train_seq_c))
        
        """Create PyTorch datasets. Testing if works better than TimeSeriesDataset."""
        from torch.utils.data import Dataset

        class PowerDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.FloatTensor(X)
                self.y = torch.FloatTensor(y)

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        train_dataset = PowerDataset(X_train_seq, y_train_seq)
        val_dataset = PowerDataset(X_val_seq, y_val_seq)
        """End of test."""
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader, X_train_seq, y_train_seq, X_val_seq, y_val_seq
    
    def extract_metadata(self, X_train_seq, X_val_seq):
        """Extract building power and demand flags for evaluation."""
        # Training metadata
        train_building_power_scaled = X_train_seq[:, -1, 2]
        temp = np.zeros((len(train_building_power_scaled), 19))
        temp[:, 2] = train_building_power_scaled
        temp_site_a = temp[:len(temp)//2]
        temp_site_c = temp[len(temp)//2:]
        train_building_power_site_a = self.data['scaler_X_site_a'].inverse_transform(temp_site_a)[:, 2]
        train_building_power_site_c = self.data['scaler_X_site_c'].inverse_transform(temp_site_c)[:, 2]
        train_building_power = np.concatenate([train_building_power_site_a, train_building_power_site_c])
        
        # Validation metadata
        val_building_power_scaled = X_val_seq[:, -1, 2]
        temp_val = np.zeros((len(val_building_power_scaled), 19))
        temp_val[:, 2] = val_building_power_scaled
        val_building_power = self.data['scaler_X_site_b'].inverse_transform(temp_val)[:, 2]
        
        # Demand flags
        train_demand_flags = np.argmax(X_train_seq[:, -1, 30:33], axis=1) - 1
        val_demand_flags = np.argmax(X_val_seq[:, -1, 30:33], axis=1) - 1
        
        # Site labels
        train_site_a = np.array(['Site A'] * len(train_building_power_site_a))
        train_site_c = np.array(['Site C'] * len(train_building_power_site_c))
        train_sites = np.concatenate([train_site_a, train_site_c])
        val_sites = np.array(['Site B'] * len(X_val_seq))
        
        return {
            'train_building_power': train_building_power,
            'val_building_power': val_building_power,
            'train_demand_flags': train_demand_flags,
            'val_demand_flags': val_demand_flags,
            'train_sites': train_sites,
            'val_sites': val_sites
        }
    
    def compute_loss_with_penalty(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        X_batch: torch.Tensor
    ):
        """
        Compute loss with false positive penalty.
        
        Args:
            outputs: Model predictions
            targets: True targets
            X_batch: Input batch (used to extract demand flags)
            
        Returns:
            Tuple of (total_loss, base_loss, penalty_term)
        """
        # Base MSE loss
        base_loss = nn.MSELoss()(outputs, targets)
        
        # Extract penalty hyperparameters
        penalty_weight = getattr(self.config, "fp_penalty_weight", 50.0)
        penalty_threshold = getattr(self.config, "fp_penalty_threshold", 0.1)
        
        # Determine "no DR" mask from demand flag one-hot encoding
        dflag_onehots = X_batch[:, -1, 30:33]  # Last timestep, DR flag columns
        dflag_ids = torch.argmax(dflag_onehots, dim=1)  # Values in {0,1,2}
        mask_no_dr = (dflag_ids == 1)  # Boolean mask for "no event"
        
        # Compute penalty only for no-DR samples
        if mask_no_dr.any():
            preds_no_dr = outputs[mask_no_dr].view(-1)
            penalty_term = torch.mean(preds_no_dr ** 2)
        else:
            penalty_term = torch.tensor(0.0, device=outputs.device)
        
        total_loss = base_loss + penalty_weight * penalty_term
        
        return total_loss, base_loss, penalty_term
    
    def train_epoch(self, model, train_loader, optimizer, epoch):
        """Train for one epoch."""
        model.train()
        train_loss = 0
        train_preds_list = []
        train_targets_list = []
        
        # Learning rate warmup
        if epoch < self.config.warmup_epochs:
            current_lr = self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs
        else:
            current_lr = self.config.learning_rate
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            # Forward pass with penalty
            outputs = model(X_batch)
            loss, base_loss, penalty_term = self.compute_loss_with_penalty(
                outputs, y_batch, X_batch
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), self.config.gradient_clip_val)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds_list.append(outputs.detach().cpu().numpy())
            train_targets_list.append(y_batch.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_preds = np.concatenate(train_preds_list)
        train_targets = np.concatenate(train_targets_list)
        
        return train_loss, train_preds, train_targets, current_lr, base_loss, penalty_term
    
    def validate(self, model, val_loader):
        """Validate the model."""
        model.eval()
        val_loss = 0
        val_preds_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = model(X_batch)
                loss = nn.MSELoss()(outputs, y_batch)
                val_loss += loss.item()
                
                val_preds_list.append(outputs.cpu().numpy())
                val_targets_list.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_preds = np.concatenate(val_preds_list)
        val_targets = np.concatenate(val_targets_list)
        
        return val_loss, val_preds, val_targets
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting run with config:")
        for key, value in dict(self.config).items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")
        
        # Prepare data
        train_loader, val_loader, X_train_seq, y_train_seq, X_val_seq, y_val_seq = self.prepare_dataloaders()
        metadata = self.extract_metadata(X_train_seq, X_val_seq)
        
        # Initialize model
        input_size = X_train_seq.shape[2]
        model = create_model(self.config, input_size).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Early stopping setup
        early_stopping_patience = 10
        best_nmae = float('inf')
        epochs_without_improvement = 0
        
        print("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, train_preds, train_targets, current_lr, base_loss, penalty_term = self.train_epoch(
                model, train_loader, optimizer, epoch
            )
            
            # Inverse transform predictions for evaluation
            train_preds_original_site_a = self.data['scaler_y_site_a'].inverse_transform(
                train_preds[:len(self.data['y_site_a'])]
            )
            train_targets_original_site_a = self.data['scaler_y_site_a'].inverse_transform(
                train_targets[:len(self.data['y_site_a'])]
            )
            train_preds_original_site_c = self.data['scaler_y_site_c'].inverse_transform(
                train_preds[len(self.data['y_site_a']):]
            )
            train_targets_original_site_c = self.data['scaler_y_site_c'].inverse_transform(
                train_targets[len(self.data['y_site_a']):]
            )
            
            train_preds_original = np.concatenate([train_preds_original_site_a, train_preds_original_site_c])
            train_targets_original = np.concatenate([train_targets_original_site_a, train_targets_original_site_c])
            
            # Calculate training metrics
            if self.evaluation_fn:
                train_metrics = self.evaluation_fn(
                    y_true=train_targets_original,
                    y_pred=train_preds_original,
                    site_labels=metadata['train_sites'],
                    building_power=metadata['train_building_power'],
                    demand_flags=metadata['train_demand_flags']
                )
            else:
                train_metrics = {}
            
            # Validate
            val_loss, val_preds, val_targets = self.validate(model, val_loader)
            
            val_preds_original = self.data['scaler_y_site_b'].inverse_transform(val_preds)
            val_targets_original = self.data['scaler_y_site_b'].inverse_transform(val_targets)
            
            if self.evaluation_fn:
                val_metrics = self.evaluation_fn(
                    y_true=val_targets_original.flatten(),
                    y_pred=val_preds_original.flatten(),
                    site_labels=metadata['val_sites'],
                    building_power=metadata['val_building_power'],
                    demand_flags=metadata['val_demand_flags']
                )
            else:
                val_metrics = {}
            
            # Log to W&B
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'learning_rate': current_lr,
                'train/base_loss': base_loss.item(),
                'train/penalty_term': penalty_term.item() if isinstance(penalty_term, torch.Tensor) else float(penalty_term),
                'train/total_loss': train_loss
            }
            
            # Add metrics if available
            for metric_name, value in train_metrics.items():
                log_dict[f'train/{metric_name}'] = value
            for metric_name, value in val_metrics.items():
                log_dict[f'val/{metric_name}'] = value
            
            wandb.log(log_dict)
            
            # Console output
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'\nEpoch [{epoch+1}/{self.config.num_epochs}]')
                print(f'Learning Rate: {current_lr:.6f}')
                print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
                if 'nmae_mean' in val_metrics:
                    print(f'Val NMAE(mean): {val_metrics["nmae_mean"]:.2f}%')
            
            # Early stopping & save best model
            current_nmae = val_metrics.get('nmae_mean', val_loss)
            if current_nmae < best_nmae:
                best_nmae = current_nmae
                epochs_without_improvement = 0
                
                # Save best model
                torch.save(model.state_dict(), self.best_model_path)
                print(f"Saved new best model at epoch {epoch+1} with Val NMAE(mean): {best_nmae:.2f}%")
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}. Best Val NMAE(mean): {best_nmae:.2f}%")
                break
        
        print(f"\nTraining completed! Best Val NMAE(mean): {best_nmae:.2f}%")
        return best_nmae
