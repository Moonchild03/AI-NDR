import pandas as pd
import numpy as np
import pickle
import joblib

# --- Load the scaler ---
with open("processed/scaler.pkl", "rb") as f:
    scaler = joblib.load(f)

# --- Load a single label encoder (used on 1 column only) ---
with open("processed/label_encoder.pkl", "rb") as f:
    single_label_encoder = joblib.load(f)

# --- Load data ---
df = pd.read_csv("multi_dataset_chunks/02-28-2018.csv", low_memory=False)
df = df.dropna()

# --- Apply single label encoder to a known column (e.g., 'Protocol') ---
col_to_encode = "Protocol"
if col_to_encode in df.columns:
    df[col_to_encode] = df[col_to_encode].astype(str)
    known_classes = set(single_label_encoder.classes_)
    df[col_to_encode] = df[col_to_encode].apply(lambda x: x if x in known_classes else "<unknown>")
    if "<unknown>" not in single_label_encoder.classes_:
        single_label_encoder.classes_ = np.append(single_label_encoder.classes_, "<unknown>")
    df[col_to_encode] = single_label_encoder.transform(df[col_to_encode])

# --- Keep only numeric data ---
df_numeric = df.select_dtypes(include=[np.number])

# --- Convert to array to avoid column name mismatch ---
X_array = df_numeric.to_numpy()

# --- Scale the data ---
X_scaled = scaler.transform(X_array)

# --- Normalize risk scores to 0–100 ---
risk_scores = np.linalg.norm(X_scaled, axis=1)
min_score = risk_scores.min()
max_score = risk_scores.max()
risk_scores_scaled = 100 * (risk_scores - min_score) / (max_score - min_score + 1e-6)

# --- Assign Risk Levels ---
def assign_risk_level(score):
    if score >= 90:
        return "Critical"
    elif score >= 75:
        return "Severe"
    elif score >= 50:
        return "High"
    elif score >= 25:
        return "Medium"
    else:
        return "Low"

risk_levels = [assign_risk_level(s) for s in risk_scores_scaled]

# --- Final Alert Table ---
alerts_df = pd.DataFrame({
    "Index": df.index,
    "Risk Score": np.round(risk_scores_scaled, 2),
    "Risk Level": risk_levels
})

top_alerts = alerts_df.sort_values(by="Risk Score", ascending=False).head(10)

print("🚨 Top Alerts (Sample):")
print(top_alerts)
