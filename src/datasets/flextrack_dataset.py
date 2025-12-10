import torch
from torch.utils.data import Dataset

class FlextrackClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)     # (N, seq_len, features)
        
        # Convert y from shape (N,1) to shape (N,) and to LongTensor
        self.y = torch.LongTensor(y.reshape(-1))   # CE requires int labels
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]