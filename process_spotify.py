import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("SpotifyFeatures.csv", low_memory=False)
print(df.head())

features = ['acousticness','danceability','duration_ms','energy','instrumentalness',
            'liveness','loudness','speechiness','tempo','valence','key','mode','time_signature']
target = ['popularity']

df = df[features + target].dropna()

if df['mode'].dtype == 'O':
    df['mode'] = df['mode'].map({'Minor': 0, 'Major': 1})

#One-hot encode the 'key' column
df = pd.get_dummies(df, columns=['key'], prefix='key', drop_first=True)

df = df.dropna(subset=features + ['popularity'])

from sklearn.preprocessing import StandardScaler

features = [col for col in df.columns if col not in ['popularity', 'track_name', 'artist_name']]

scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df[target].values

#Train/test split
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

#Top 10 genres
top10_genres = df['genre'].value_counts().head(10).index
df_top10 = df[df['genre'].isin(top10_genres)]

plt.figure(figsize=(12,6))
sns.boxplot(x='genre', y='popularity', data=df_top10, palette='Set3')
plt.title("Popularity Distribution by Top 10 Genres")
plt.xticks(rotation=45, ha='right')
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(
    x='danceability',
    y='energy',
    hue='genre',
    data=df[df['genre'].isin(top10_genres)],
    alpha=0.6
)
plt.title("Danceability vs Energy (Top 10 Genres)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

out = Path("figures"); out.mkdir(exist_ok=True)

#Popularity distribution
plt.figure(figsize=(8,5))
plt.hist(df["popularity"], bins=100)
plt.title("Popularity Distribution (100 bins)")
plt.xlabel("Popularity"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(out/"popularity_hist_100.png"); plt.close()

plt.figure(figsize=(8,5))
plt.hist(df["tempo"].dropna(), bins=60)
plt.title("Tempo Distribution")
plt.xlabel("Tempo (BPM)"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(out/"tempo_hist.png"); plt.close()

#Danceability vs Energy
plt.figure(figsize=(6,6))
hb = plt.hexbin(df["danceability"], df["energy"], gridsize=40, mincnt=1)
plt.title("Danceability vs Energy")
plt.xlabel("Danceability"); plt.ylabel("Energy")
cb = plt.colorbar(hb); cb.set_label("Count")
plt.tight_layout(); plt.savefig(out/"danceability_vs_energy_hex.png"); plt.close()

#Correlation heatmap
num_cols = ["popularity","acousticness","danceability","duration_ms","energy",
            "instrumentalness","liveness","loudness","speechiness","tempo",
            "valence","key","mode","time_signature"]
num_cols = [c for c in num_cols if c in df.columns]
corr = df[num_cols].corr()

plt.figure(figsize=(10,8))
plt.imshow(corr, aspect="auto")
plt.title("Correlation Heatmap (numeric features)")
plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
plt.yticks(range(len(num_cols)), num_cols)
plt.colorbar(label="corr")
plt.tight_layout(); plt.savefig(out/"correlation_heatmap.png"); plt.close()

# Popularity vs Valence with simple trendline
x = df["valence"].to_numpy()
y = df["popularity"].to_numpy()
mask = ~np.isnan(x) & ~np.isnan(y)
x, y = x[mask], y[mask]

plt.figure(figsize=(8,6))
plt.scatter(x, y, alpha=0.3, s=8)
if len(x) > 1:
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(0, 1, 100)
    plt.plot(xs, m*xs + b, linestyle="--")
plt.title("Popularity vs Valence (with trendline)")
plt.xlabel("Valence"); plt.ylabel("Popularity")
plt.tight_layout(); plt.savefig(out/"popularity_vs_valence.png"); plt.close()

print("Saved figures to:", out.resolve())