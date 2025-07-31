import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set display and style
sns.set(style="whitegrid")

# Paths
raw_file_path = 'multi_dataset_chunks/1.csv'  # choose any raw file
processed_file_path = 'processed/final_combined_dataset.csv'

# Load a sample raw dataset
raw_df = pd.read_csv(raw_file_path, low_memory=False)

# Load processed dataset
processed_df = pd.read_csv(processed_file_path)

# --- 1. Compare number of features ---
print(f"Raw feature count: {raw_df.shape[1]}")
print(f"Processed feature count: {processed_df.shape[1]}")

plt.figure(figsize=(6, 4))
plt.bar(["Raw", "Processed"], [raw_df.shape[1], processed_df.shape[1]], color=['skyblue', 'lightgreen'])
plt.title("📊 Feature Count: Raw vs Processed")
plt.ylabel("Number of Features")
plt.tight_layout()
plt.show()

# --- 2. Compare label distributions ---
raw_label_dist = raw_df['Label'].value_counts().head(10)  # top 10 only
processed_label_dist = processed_df['Label'].value_counts()

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(x=raw_label_dist.values, y=raw_label_dist.index, ax=axs[0])
axs[0].set_title("🧪 Raw Label Distribution")
axs[0].set_xlabel("Count")

sns.barplot(x=processed_label_dist.values, y=processed_label_dist.index, ax=axs[1])
axs[1].set_title("✅ Processed Label Distribution")
axs[1].set_xlabel("Count")

plt.tight_layout()
plt.show()

# --- 3. Show distribution of one key feature before/after ---
common_features = ['Destination Port', 'Flow Duration', 'Total Fwd Packets']
for feature in common_features:
    if feature in raw_df.columns and feature in processed_df.columns:
        plt.figure(figsize=(10, 4))
        sns.kdeplot(raw_df[feature].dropna(), label='Raw', fill=True)
        sns.kdeplot(processed_df[feature].dropna(), label='Processed', fill=True)
        plt.title(f"📈 Distribution Before vs After: {feature}")
        plt.legend()
        plt.tight_layout()
        plt.show()
