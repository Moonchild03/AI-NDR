import os
import pandas as pd

def check_label_distribution(folder_path="multi_dataset_chunks"):
    print(f"\n📊 Checking label distribution in each CSV in '{folder_path}':\n")

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            try:
                df = pd.read_csv(file_path)
                if "Label" in df.columns:
                    print(f"📂 {file_name}")
                    print(df["Label"].value_counts(dropna=False, normalize=False))
                    print("-" * 40)
                else:
                    print(f"⚠️  {file_name} - 'Label' column not found.")
            except Exception as e:
                print(f"❌ Failed to read {file_name}: {e}")

if __name__ == "__main__":
    check_label_distribution()
