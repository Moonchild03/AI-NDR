import pandas as pd
import os

# Paths to processed X and y
x_path = "processed/x.csv"
y_path = "processed/y.csv"

# Output path
output_path = "processed/final_combined_dataset.csv"

# Load the feature and label data
print("📥 Loading processed X and y...")
X = pd.read_csv(x_path)
y = pd.read_csv(y_path, names=["Label"])  # Ensure label column is named

# Handle length mismatch
if len(X) != len(y):
    print(f"⚠️ Length mismatch: X={len(X)}, y={len(y)} — trimming to match...")
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len].reset_index(drop=True)
    y = y.iloc[:min_len].reset_index(drop=True)

# Combine into one DataFrame
print("🛠️ Combining features and labels...")
df_combined = pd.concat([X, y], axis=1)

# Save to CSV
df_combined.to_csv(output_path, index=False)
print(f"✅ Combined dataset saved at: {output_path}")
