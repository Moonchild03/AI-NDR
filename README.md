# AI-NDR
# 🛡️ AI-NDR: Network Detection & Response with Risk Scoring

This project is an AI-powered Network Detection and Response (NDR) system designed to analyze network traffic, classify potential threats, and assign a prioritized risk score to generate actionable alerts. Built using a Random Forest model trained on CICIDS2018 datasets, it automatically preprocesses new data, applies encodings and scaling, and outputs human-readable risk levels such as **Critical**, **Severe**, **High**, **Medium**, and **Low**.

## 🚀 Features

- ✅ Network traffic classification using trained ML model
- ✅ Automatic risk scoring normalized between 0–100
- ✅ Prioritized alert generation with readable risk levels
- ✅ Scalable pipeline for integration into SOC/SIEM workflows

## 📁 Structure

AI-IDS/
├── data/ # Input network data (e.g., CICIDS .csv files)
├── models/
│ ├── random_forest_model.pkl # Trained Random Forest model
│ ├── scaler.pkl # StandardScaler used during training
│ └── label_encoder.pkl # Label encoders for categorical columns
├── scripts/
│ ├── final_risk.py # Main script for scoring and alert generation
│ └── train_model.py # (Optional) Script for model training
├── requirements.txt # Python dependencies
└── README.md # Project documentation
