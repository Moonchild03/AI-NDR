🛡️ AI-NDR: Network Detection & Response with Risk Scoring
AI-NDR is an AI-powered Network Detection and Response (NDR) system designed to intelligently monitor network traffic, classify malicious behavior, and assign prioritized risk scores. It generates human-readable alerts to support faster, smarter incident response — ideal for integration into Security Operations Centers (SOC) and SIEM pipelines.

**🚀 Key Features**
✅ Real-Time Traffic Classification
Powered by a trained Random Forest model, the system can detect and classify network threats using deep feature analysis from CICIDS2018 dataset.

✅ Automated Risk Scoring
Assigns dynamic risk scores (0–100) to each detected event based on prediction confidence, threat type, and context (e.g., destination sensitivity).

✅ Prioritized Alerting System
Generates actionable alerts categorized as:

🔥 Critical (≥ 90)

⚠️ Severe (80–89)

⚠️ High (70–79)

⚠️ Medium (50–69)

✅ Low (< 50)

✅ Modular and Scalable Design
Built as a pipeline with clearly separated components — model inference, scoring logic, alert generation — making it easy to extend or integrate with external monitoring tools.

✅ SOC/SIEM Integration Ready
Alert output is structured and easily streamable to platforms like Splunk, Wazuh, Elastic (ELK), Graylog, or even Slack/Email for rapid visibility.

**🧠 How It Works**
Data Ingestion
Network traffic logs (e.g., CICIDS2018 .csv) are fed into the system either in batch or real-time stream format.

Preprocessing Pipeline

Categorical encoding (e.g., protocol type, service)

Numerical scaling (using StandardScaler)

Feature alignment to match training schema

Threat Classification
The preprocessed data is passed through a trained Random Forest model that outputs a prediction (Benign, DoS, Probe, etc.) and confidence score.

Risk Scoring
A dynamic risk score is computed using:

Prediction confidence

Threat severity

Asset criticality (customizable IP/IP ranges)

TI correlation (optional)

Alert Generation
Each event is assigned a severity level based on risk score and logged or sent to external systems for analyst review.

**🧱 Project Structure**
AI-NDR/
│
├── data/                    # Input data directory (e.g., CICIDS2018 .csv files)
│
├── models/                  # Pretrained models and transformers
│   ├── random_forest_model.pkl     # Trained Random Forest classifier
│   ├── scaler.pkl                  # StandardScaler used for numerical normalization
│   └── label_encoder.pkl           # Encoders for categorical columns (protocols, services)
│
├── scripts/
│   ├── train_model.py              # (Optional) Script to retrain model with custom dataset
│   └── final_risk.py               # Core NDR script: classification + risk scoring + alerting
│
├── requirements.txt        # Python package dependencies
│
└── README.md               # Project documentation (this file)

**Technologies Used**
Python 3.10+
scikit-learn, pandas, numpy
joblib for model serialization
matplotlib and seaborn (for model evaluation)
FastAPI (optional, for real-time serving)
Dataset: CICIDS 2018

