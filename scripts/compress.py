import joblib

# Load the current model
with open("models/random_forest_model.pkl", "rb") as f:
    model = joblib.load(f)

# Save it compressed
joblib.dump(model, "random_forest_model_compressed.pkl", compress=3)
