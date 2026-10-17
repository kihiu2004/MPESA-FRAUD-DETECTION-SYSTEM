from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.features.behavioral_features import create_behavioral_features
from src.features.transaction_features import create_transaction_features, resolve_column

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
ANOMALY_MODEL_PATH = MODEL_DIR / "anomaly_model.pkl"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_transactions.csv"


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = [
        "step",
        "type",
        "amount",
        "nameorig",
        "oldbalanceorg",
        "newbalanceorig",
        "namedest",
        "oldbalancedest",
        "newbalancedest",
        "isfraud",
        "isflaggedfraud",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    ]
    return [col for col in df.columns if col.lower() not in exclude and col not in exclude]


def engineer_features(
    df: pd.DataFrame,
    large_amount_threshold: float = None,
) -> pd.DataFrame:
    df_t = create_transaction_features(df, large_amount_threshold=large_amount_threshold)
    df_b = create_behavioral_features(df_t)
    return df_b


def train_anomaly_detector(
    df: pd.DataFrame,
    contamination: float = 0.001,
    random_state: int = 42,
) -> Dict[str, Any]:
    # Persist the amount threshold so detect_anomalies() reproduces the exact
    # same is_large_transaction feature used while fitting.
    amount_col = resolve_column(df, "amount")
    threshold = (
        pd.to_numeric(df[amount_col], errors="coerce").median()
        if amount_col is not None
        else None
    )
    df_eng = engineer_features(df.copy(), large_amount_threshold=threshold)
    feature_cols = get_feature_columns(df_eng)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_eng[feature_cols].fillna(0))
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "contamination": contamination,
        "large_amount_threshold": threshold,
    }


def save_anomaly_model(bundle: Dict[str, Any]) -> str:
    import pickle

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANOMALY_MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return str(ANOMALY_MODEL_PATH)


def load_anomaly_model() -> Dict[str, Any]:
    import pickle

    if not ANOMALY_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {ANOMALY_MODEL_PATH}")
    with open(ANOMALY_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def detect_anomalies(df: pd.DataFrame, bundle: Dict[str, Any] = None) -> pd.DataFrame:
    if bundle is None:
        bundle = load_anomaly_model()
    threshold = bundle.get("large_amount_threshold")
    df_eng = engineer_features(df.copy(), large_amount_threshold=threshold)
    scaler = bundle["scaler"]
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    for c in feature_cols:
        if c not in df_eng.columns:
            df_eng[c] = 0
    X_scaled = scaler.transform(df_eng[feature_cols].fillna(0))
    scores = model.score_samples(X_scaled)
    preds = model.predict(X_scaled)
    result = df.copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = (preds == -1).astype(int)
    return result


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DATA_PATH, nrows=10000)
    bundle = train_anomaly_detector(df)
    save_anomaly_model(bundle)
    res = detect_anomalies(df.head(100), bundle)
    print(res[["anomaly_score", "is_anomaly"]].head())
