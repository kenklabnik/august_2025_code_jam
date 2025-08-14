import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("SpotifyFeatures.csv", low_memory=False)

print(df.head())

features = ['acousticness','danceability','duration_ms','energy','instrumentalness',
            'liveness','loudness','speechiness','tempo','valence','key','mode','time_signature']
target = ['popularity']

df = df[features + target].dropna()


scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Dataset Class
class SpotifyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_data = SpotifyDataset(X_train, y_train)
test_data = SpotifyDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
