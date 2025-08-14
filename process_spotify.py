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


class SpotifyNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = SpotifyNet(input_size=len(features))

#Training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

#Evaluation
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch)
        predictions.extend(preds.numpy().flatten())
        actuals.extend(y_batch.numpy().flatten())

predictions = np.array(predictions)
actuals = np.array(actuals)

rmse = np.sqrt(np.mean((predictions - actuals)**2))
mae = np.mean(np.abs(predictions - actuals))
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

#Plot
plt.scatter(actuals, predictions, alpha=0.5)
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Predicted vs Actual Popularity")
plt.plot([0, 100], [0, 100], '--', color='red')
plt.show()

