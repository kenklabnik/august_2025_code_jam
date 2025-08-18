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
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader


df_plot = pd.read_csv("SpotifyFeatures.csv", low_memory=False)

df = pd.read_csv("SpotifyFeatures.csv", low_memory=False)
print(df.head())

base_features = [
    'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'key', 'mode', 'time_signature']
target = 'popularity'

df = df[base_features + [target]].copy()

if df['mode'].dtype == 'O':
    df['mode'] = df['mode'].map({'Minor': 0, 'Major': 1})

df = pd.get_dummies(df, columns=['key'], prefix='key', drop_first=True)

df['time_signature'] = df['time_signature'].astype(str).str.split('/').str[0]
df['time_signature'] = pd.to_numeric(df['time_signature'], errors='coerce')

# Build final feature list
features = [c for c in df.columns if c != target]

df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df[features] = df[features].fillna(df[features].median(numeric_only=True))
df[target] = pd.to_numeric(df[target], errors='coerce').fillna(df[target].median())

before = len(df)
df = df.dropna(subset=features + [target])
after = len(df)
print(f"Rows after cleaning: {after} (dropped {before - after})")

# Scale + split
scaler = StandardScaler()
X = scaler.fit_transform(df[features].values)
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test.reshape(-1, 1),  dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=64, shuffle=False)

# Model
class SpotifyNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))
    def forward(self, x): return self.net(x)

model = SpotifyNet(input_size=X_train.shape[1]).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 20

for epoch in range(epochs):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running += loss.item()
    print(f"Epoch {epoch+1}/{epochs} | train MSE: {running/len(train_loader):.4f}")

model.eval()
preds, acts = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out = model(xb).cpu().numpy().flatten()
        preds.extend(out)
        acts.extend(yb.numpy().flatten())

preds = np.array(preds)
acts = np.array(acts)
rmse = np.sqrt(((preds - acts) ** 2).mean())
mae = np.abs(preds - acts).mean()
print(f"Test RMSE: {rmse:.2f} | MAE: {mae:.2f}")

# Save model + scaler
torch.save(model.state_dict(), "spotifynet.pt")
import joblib; joblib.dump(scaler, "scaler.joblib")
print("Saved model -> spotifynet.pt; scaler -> scaler.joblib")

out = Path("figures"); out.mkdir(exist_ok=True)

# Pred vs Actual
plt.figure(figsize=(6,6))
plt.scatter(acts, preds, alpha=0.4, s=8)
plt.plot([0,100],[0,100],'--')
plt.xlabel("Actual Popularity"); plt.ylabel("Predicted Popularity")
plt.title("Predicted vs Actual Popularity")
plt.tight_layout(); plt.savefig(out / "pred_vs_actual.png", dpi=150); plt.close()

top10_genres = df_plot['genre'].dropna().value_counts().head(10).index
df_top10 = df_plot[df_plot['genre'].isin(top10_genres)]

plt.figure(figsize=(12,6))
sns.boxplot(x='genre', y='popularity', hue='genre', data=df_top10, palette='Set3', legend=False)
plt.title("Popularity Distribution by Top 10 Genres")
plt.xticks(rotation=45, ha='right')
plt.tight_layout(); plt.savefig(out / "popularity_box_by_genre.png", dpi=150); plt.close()

# Danceability vs Energy colored by genre (Seaborn)
plt.figure(figsize=(8,6))
sns.scatterplot(
    x='danceability', y='energy',
    hue='genre',
    data=df_plot[df_plot['genre'].isin(top10_genres)], alpha=0.6, legend='brief')
plt.title("Danceability vs Energy (Top 10 Genres)")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout(); plt.savefig(out / "dance_vs_energy_by_genre.png", dpi=150); plt.close()

# Popularity distribution
plt.figure(figsize=(8,5))
plt.hist(df["popularity"], bins=100)
plt.title("Popularity Distribution (100 bins)")
plt.xlabel("Popularity"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(out / "popularity_hist_100.png", dpi=150); plt.close()

# Tempo distribution
plt.figure(figsize=(8,5))
plt.hist(df["tempo"].dropna(), bins=60)
plt.title("Tempo Distribution")
plt.xlabel("Tempo (BPM)"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(out / "tempo_hist.png", dpi=150); plt.close()

# Danceability vs Energy (hexbin)
plt.figure(figsize=(6,6))
hb = plt.hexbin(df["danceability"], df["energy"], gridsize=40, mincnt=1)
plt.title("Danceability vs Energy")
plt.xlabel("Danceability"); plt.ylabel("Energy")
cb = plt.colorbar(hb); cb.set_label("Count")
plt.tight_layout(); plt.savefig(out / "danceability_vs_energy_hex.png", dpi=150); plt.close()

# Correlation heatmap
heat_cols = [
    "popularity","acousticness","danceability","duration_ms","energy",
    "instrumentalness","liveness","loudness","speechiness","tempo",
    "valence","mode","time_signature"]
heat_cols = [c for c in heat_cols if c in df.columns]
corr = df[heat_cols].corr()

plt.figure(figsize=(10,8))
plt.imshow(corr, aspect="auto")
plt.title("Correlation Heatmap (numeric features)")
plt.xticks(range(len(heat_cols)), heat_cols, rotation=45, ha="right")
plt.yticks(range(len(heat_cols)), heat_cols)
plt.colorbar(label="corr")
plt.tight_layout(); plt.savefig(out / "correlation_heatmap.png", dpi=150); plt.close()

print("Saved figures to:", out.resolve())

plt.figure(figsize=(6,6))
hb = plt.hexbin(df["loudness"], df["energy"], gridsize=40, mincnt=1)
plt.title("Loudness vs Energy")
plt.xlabel("Loudness (dB)")
plt.ylabel("Energy")
cb = plt.colorbar(hb); cb.set_label("Count")
plt.tight_layout(); plt.savefig(out / "loudness_vs_energy.png", dpi=150); plt.close()

avg_pop_by_key = df_plot.groupby("key")["popularity"].mean().sort_values()
plt.figure(figsize=(10,5))
avg_pop_by_key.plot(kind="bar", color="skyblue")
plt.title("Average Popularity by Musical Key")
plt.xlabel("Key")
plt.ylabel("Avg Popularity")
plt.tight_layout(); plt.savefig(out / "avg_popularity_by_key.png", dpi=150); plt.close()

