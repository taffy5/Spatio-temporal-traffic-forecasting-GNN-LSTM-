import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

SEQ_LEN = 12      # past 1 hour
PRED_LEN = 3      # next 15 mins


class TrafficDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        self.scaler = StandardScaler()
        data = self.scaler.fit_transform(df.values)

        self.X, self.y = self.create_sequences(data)

    def create_sequences(self, data):
        xs, ys = [], []
        for i in range(len(data) - SEQ_LEN - PRED_LEN):
            xs.append(data[i:i+SEQ_LEN])
            ys.append(data[i+SEQ_LEN:i+SEQ_LEN+PRED_LEN])
        return torch.FloatTensor(xs), torch.FloatTensor(ys)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
