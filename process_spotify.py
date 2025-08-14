import pandas as pd

# Load the CSV
df = pd.read_csv("SpotifyFeatures.csv", low_memory=False)

# Show basic info
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())