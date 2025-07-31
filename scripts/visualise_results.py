import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Load model and dataset
model = joblib.load("models/random_forest_model.pkl")
df = pd.read_csv("processed/final_combined_dataset.csv")

# Separate features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Train-test split (same as training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Predict and create confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# 🔷 Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y.unique()),
            yticklabels=sorted(y.unique()))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()

# 🔷 Plot feature importances
importances = model.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.show()
