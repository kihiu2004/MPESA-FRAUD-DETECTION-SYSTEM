# M-Pesa Fraud Detection System

A comprehensive fraud detection system for M-Pesa mobile money transactions using machine learning, rule-based detection, and anomaly detection.

## Overview

This project implements a multi-layered fraud detection approach that combines:
- **Machine Learning Models**: Random Forest, XGBoost, and Logistic Regression classifiers
- **Rule-Based Detection**: Expert-defined rules based on transaction patterns
- **Anomaly Detection**: Isolation Forest for identifying unusual transaction behaviors
- **Risk Scoring**: Combined risk assessment with explainable reasons

## Project Structure

```
M_PESA FRAUD DETECTION SYSTEM/
├── data/
│   ├── external/          # External data sources
│   ├── processed/         # Cleaned and processed data
│   └── raw/               # Raw data files
├── dashboard/
│   └── streamlit_app.py   # Streamlit web application
├── models/                # Trained model files
├── notebooks/             # Jupyter notebooks for analysis
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── detection/         # Fraud detection models
│   ├── evaluation/        # Model evaluation
│   ├── features/          # Feature engineering
│   └── utils/             # Utility functions
├── main.py               # Main pipeline script
└── requirements.txt      # Project dependencies
```

## Key Features

- **Data Preprocessing**: Automated cleaning and validation of transaction data
- **Feature Engineering**: Transaction-level and behavioral features
- **Multi-Model Detection**: Combines ML, rules, and anomaly detection
- **Risk Scoring**: Calculates comprehensive risk scores with explanations
- **Interactive Dashboard**: Streamlit app for real-time analysis
- **Batch Processing**: Analyze multiple transactions at once

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the complete pipeline:
```bash
python main.py
```
   - Results are cached between runs: an existing `data/processed/cleaned_transactions.csv`
     and trained models are reused.
   - Pass `--refresh` to force rebuilding the cleaned dataset from the raw PaySim file:
   ```bash
   python main.py --refresh
   ```
   - The pipeline is memory-bounded: the raw file is streamed in chunks and models are
     trained on a capped sample, so it runs on a machine with ~8 GB of RAM.

4. Launch the dashboard:
```bash
streamlit run dashboard/streamlit_app.py
```

## Usage

### Single Transaction Analysis
Use the dashboard to analyze individual transactions by entering transaction details.

### Batch Analysis
Upload CSV files with transaction data for bulk fraud detection.

### API Usage
```python
from src.utils.risk_scoring import combine_risk_scores
import pandas as pd

df = pd.DataFrame({...})
results = combine_risk_scores(df)
```

## Models

- **Fraud Classifier**: Supervised learning model trained on historical fraud data
- **Anomaly Detector**: Unsupervised Isolation Forest for outlier detection
- **Rule Engine**: Heuristic rules for known fraud patterns

## Evaluation Metrics

The system is evaluated using standard metrics:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

## Dataset

Based on PaySim synthetic mobile money transaction dataset, simulating real M-Pesa transaction patterns.

## License

MIT License
