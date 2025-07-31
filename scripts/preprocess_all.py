import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Paths
data_dir = os.path.join(os.getcwd(), "multi_dataset_chunks")
output_dir = os.path.join(os.getcwd(), "processed")
os.makedirs(output_dir, exist_ok=True)

label_map = {
    "Benign": "Benign",
    "DDOS attack-HOIC": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS",
    "DoS attacks-Hulk": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS",
    "Infilteration": "Infiltration",
    "Bot": "Bot",
}

X_list, y_list = [], []
le = LabelEncoder()
scaler = StandardScaler()

def clean_chunk(df):
    # Drop junk rows where Label column literally has value "Label"
    if "Label" not in df.columns:
        raise ValueError("Missing 'Label' column in input data")
    df = df[df["Label"] != "Label"]

    # Drop extra columns if present
    for col in ["Flow ID", "Timestamp", "Attempted Category"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Map labels
    df["Label"] = df["Label"].map(label_map).dropna()

    # Separate features and labels
    y = df["Label"]
    X = df.drop("Label", axis=1)

    # Encode object columns
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category").cat.codes

    # Replace bad values
    X.replace([float('inf'), float('-inf')], 0, inplace=True)
    X.fillna(0, inplace=True)

    return X, y

print("📦 Processing files in 'multi_dataset_chunks'...\n")

for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(data_dir, file)
        print(f" {file}")
        try:
            df = pd.read_csv(file_path, low_memory=False)
            X, y = clean_chunk(df)
            X_list.append(X)
            y_list.append(y)
        except Exception as e:
            print(f"Error processing {file}: {e}")
        print("-" * 40)

# Combine and encode
X_all = pd.concat(X_list, ignore_index=True)
y_all = pd.concat(y_list, ignore_index=True)
y_all = le.fit_transform(y_all)

# Scale features
X_scaled = scaler.fit_transform(X_all)

# Save processed data
pd.DataFrame(X_scaled).to_csv(os.path.join(output_dir, "X.csv"), index=False)
pd.DataFrame(y_all).to_csv(os.path.join(output_dir, "y.csv"), index=False)

# Save scaler, label encoder, and column names
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))

with open(os.path.join(output_dir, "columns.txt"), "w") as f:
    for col in X_all.columns:
        f.write(col + "\n")

print(f"\n Saved processed files to: {output_dir}")
print(" Saved scaler, label encoder, and column names to 'processed/'")
