import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

print("🚀 Starting model training...")

# Load processed data
X_path = "processed/X.csv"
y_path = "processed/y.csv"

if not os.path.exists(X_path) or not os.path.exists(y_path):
    raise FileNotFoundError("❌ X.csv or y.csv not found in /processed folder.")

X = pd.read_csv(X_path)
y = pd.read_csv(y_path).squeeze()

print(f"✅ Loaded data: {X.shape[0]} rows, {X.shape[1]} features")

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Compute class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, class_weights))
print(f"⚖️  Computed class weights: {class_weights_dict}")

# Train model
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight=class_weights_dict)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n📊 Evaluation Results:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and transformers
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/random_forest_model.pkl")
print("✅ Model saved: models/random_forest_model.pkl")

# Save scaler & encoder
scaler = StandardScaler().fit(X)
encoder = LabelEncoder().fit(y)

joblib.dump(scaler, "processed/scaler.pkl")
joblib.dump(encoder, "processed/label_encoder.pkl")
print("✅ Scaler & LabelEncoder saved to processed/")
