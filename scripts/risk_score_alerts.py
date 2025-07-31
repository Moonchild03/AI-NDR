import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Load saved objects
scaler = joblib.load("processed/scaler.pkl")
label_encoder = joblib.load("processed/label_encoder.pkl")

# Load the dataset (change path if needed)
df = pd.read_csv("multi_dataset_chunks/02-28-2018.csv", low_memory=False)

# Drop non-numeric columns or encode them
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = label_encoder.fit_transform(df[col].astype(str))
        except:
            df[col] = pd.factorize(df[col])[0]  # fallback encoding

# Replace infinity and NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Align columns with training data
# Assuming the training data had 79 features named 0, 1, 2, ..., 78
# Rename columns to match training time (numeric)
df.columns = [str(i) for i in range(len(df.columns))]

# Filter only the columns that the scaler was trained on
expected_cols = scaler.feature_names_in_
df = df[expected_cols]  # Select only those columns

X_scaled = scaler.transform(df)

# Generate risk scores (Euclidean distance from origin)
risk_scores = np.linalg.norm(X_scaled, axis=1)

# --- Normalize risk scores to 0-100 range ---
risk_scores = np.linalg.norm(X_scaled, axis=1)

# Scale to 0–100
min_score = risk_scores.min()
max_score = risk_scores.max()
risk_scores_scaled = 100 * (risk_scores - min_score) / (max_score - min_score)

# Assign Risk Levels
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


# Create alerts
alerts = []
for i, score in enumerate(risk_scores):
    alerts.append({
        "Index": i,
        "Risk Score": round(score, 2),
        "Risk Level": assign_risk_level(score)
    })

alerts_df = pd.DataFrame({
    'Index': df.index,
    'Risk Score': risk_scores_scaled,
    'Risk Level': [assign_risk_level(score) for score in risk_scores_scaled]
})

alerts_df = alerts_df.sort_values(by="Risk Score", ascending=False)
print("🚨 Top Alerts (Sample):")
print(alerts_df.head(10))

