import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# 🔧 Paths
MODEL_PATH = "models/random_forest_model.pkl"
SCALER_PATH = "processed/scaler.pkl"
ENCODER_PATH = "processed/label_encoder.pkl"
NEW_DATA_PATH = "multi_dataset_chunks/1.csv"  # Adjust path if needed

print("🔍 Loading model and encoders...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

print("📥 Reading new data...")
new_df = pd.read_csv(NEW_DATA_PATH, low_memory=False)  # ✅ Keep header

# 🧼 Clean + Preprocess new data
def preprocess_new_data(df):
    # Drop irrelevant columns if present
    drop_cols = ["Flow ID", "Timestamp", "Label", "Attack", "Attack category"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors="ignore")

    # Convert categorical/object columns to numeric codes
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    # Handle missing or infinite values
    df.replace([float("inf"), float("-inf")], 0, inplace=True)
    df.fillna(0, inplace=True)

    # ✅ Ensure column order matches the trained scaler
    df = df[scaler.feature_names_in_]

    # Scale features
    X_scaled = scaler.transform(df)

    return X_scaled

print("⚙️ Preprocessing new data...")
X_new = preprocess_new_data(new_df)

print("🧠 Predicting...")
y_pred = model.predict(X_new)
decoded_preds = label_encoder.inverse_transform(y_pred)

print("\n🚨 Predictions on New Data (Sample):")
print(pd.Series(decoded_preds).value_counts())

# Optional: Alert logic (flag anything not benign)
print("\n🔔 Alerts:")
for i, label in enumerate(decoded_preds):
    if label != "Benign":
        print(f"[ALERT] Row {i} flagged as: {label}")
