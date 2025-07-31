# check_final_label_distribution.py

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Load the encoded labels
y_path = "processed/y.csv"
y = pd.read_csv(y_path, header=None).squeeze()

# Count and show label distribution
counts = Counter(y)
print("📊 Final Label Distribution (Encoded):")
for label, count in counts.items():
    print(f"Class {label}: {count} samples")

# Optional: plot bar chart
plt.figure(figsize=(8,5))
plt.bar(counts.keys(), counts.values(), color='skyblue')
plt.xlabel("Encoded Class Labels")
plt.ylabel("Sample Count")
plt.title("Final Label Distribution")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(list(counts.keys()))
plt.tight_layout()
plt.show()
