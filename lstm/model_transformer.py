import pandas as pd
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=1, num_layers=4):
        super(TransformerModel, self).__init__()
        
        # 定義位置編碼
        self.pos_encoder = nn.Embedding(512, input_size)  # 假設最大長度為500
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        # 假設 x 的形狀是 [batch_size, seq_len, feature_size]
        batch_size, seq_len, feature_size = x.size()
        
        # 增加位置編碼
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        x = x + self.pos_encoder(pos)
        
        # 變換為適合 Transformer 的形狀 [seq_len, batch_size, feature_size]
        x = x.permute(1, 0, 2)
        
        # Transformer 模型
        x = self.transformer_encoder(x)
        
        # 取序列最後一步的輸出進行分類
        x = x[-1, :, :]
        
        # 全連接層
        out = self.fc(x)
        
        return out