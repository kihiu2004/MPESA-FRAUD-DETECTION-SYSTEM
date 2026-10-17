import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features.behavioral_features import create_behavioral_features
from src.features.transaction_features import create_transaction_features, resolve_column


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "fraud_model.pkl"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_transactions.csv"

# PaySim has 6.3M+ rows which does not fit in RAM on every machine. When no
# DataFrame is supplied we train on this many rows sampled from the full
# cleaned dataset, which keeps peak memory bounded while still learning from
# the whole population rather than an arbitrary handful of rows.
DEFAULT_MAX_TRAINING_ROWS = 500_000


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


def load_preprocessed_data() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df = df.copy()
    target_col = None
    for t in ["isfraud", "isFraud"]:
        if t in df.columns:
            target_col = t
            break
    if target_col is None:
        raise ValueError("Target column 'isfraud' not found in data")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def sample_processed_data(
    max_rows: int = DEFAULT_MAX_TRAINING_ROWS,
    random_state: int = 42,
    chunk_size: int = 250_000,
) -> pd.DataFrame:
    """Return a memory-bounded random sample of the processed dataset.

    The full cleaned dataset is streamed one chunk at a time so the whole
    file is never held in memory. The returned rows keep their ORIGINAL
    positional index, which is what ``build_and_save_model`` stores as
    ``test_index``; keeping it means the held-out rows still refer to the same
    rows of the full CSV (see ``evaluate_on_test_set``).
    """
    # Discover the target column without loading the data.
    header = pd.read_csv(PROCESSED_DATA_PATH, nrows=0)
    target_col = next((c for c in ("isfraud", "isFraud") if c in header.columns), None)
    if target_col is None:
        raise ValueError("Target column 'isfraud' not found in data")

    total = sum(1 for _ in open(PROCESSED_DATA_PATH, "rb")) - 1
    if max_rows is None or max_rows >= total:
        return pd.read_csv(PROCESSED_DATA_PATH)

    rng = np.random.default_rng(random_state)
    keep_probability = max_rows / total

    parts = []
    offset = 0
    for chunk in pd.read_csv(PROCESSED_DATA_PATH, chunksize=chunk_size):
        chunk.index = np.arange(offset, offset + len(chunk))
        offset += len(chunk)
        keep = rng.random(len(chunk)) < keep_probability
        if keep.any():
            parts.append(chunk[keep])

    df = pd.concat(parts)
    print(
        f"Sampled {len(df)} of {total} rows "
        f"({int(df[target_col].sum())} fraud cases) for training."
    )
    return df


def engineer_features(
    df: pd.DataFrame,
    large_amount_threshold: float = None,
) -> pd.DataFrame:
    """Build the model features for ``df``.

    ``large_amount_threshold`` must be the value fitted on the TRAINING rows
    and reused here so ``is_large_transaction`` is identical during training,
    evaluation and inference. When it is None the threshold falls back to the
    median of ``df`` (``create_transaction_features`` warns about this) which
    would make the feature depend on the rows being scored.
    """
    df_t = create_transaction_features(df, large_amount_threshold=large_amount_threshold)
    df_b = create_behavioral_features(df_t)
    return df_b


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
) -> Dict[str, Any]:
    X_train = X_train.copy()
    feature_cols = get_feature_columns(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[feature_cols])

    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X_train_scaled, y_train)
    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "model_type": model_type,
    }


def evaluate_model(
    model_bundle: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    feature_cols = model_bundle["feature_cols"]
    X_test_scaled = scaler.transform(X_test[feature_cols])
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def save_model(model_bundle: Dict[str, Any]) -> str:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f)
    return str(MODEL_PATH)


def load_model() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(df: pd.DataFrame, model_bundle: Dict[str, Any] = None) -> pd.DataFrame:
    if model_bundle is None:
        model_bundle = load_model()
    # Reuse the threshold persisted at training time so the engineered features
    # match the ones the model was fitted on (single-row requests previously
    # always produced is_large_transaction = 0).
    threshold = model_bundle.get("large_amount_threshold")
    df_feat = engineer_features(df.copy(), large_amount_threshold=threshold)
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    feature_cols = model_bundle["feature_cols"]
    missing = [c for c in feature_cols if c not in df_feat.columns]
    for c in missing:
        df_feat[c] = 0
    X_scaled = scaler.transform(df_feat[feature_cols])
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else np.array(y_pred)
    result = df.copy()
    result["fraud_prediction"] = y_pred
    result["fraud_probability"] = y_proba
    return result


def build_and_save_model(df: pd.DataFrame = None, model_type: str = "random_forest") -> str:
    if df is None:
        df = sample_processed_data()

    target_columns = [c for c in ["isfraud", "isFraud"] if c in df.columns]
    if not target_columns:
        raise ValueError("Target column 'isfraud' not found in data")
    target_col = target_columns[0]

    X = df.drop(columns=[target_col])
    # Keep the caller's index on BOTH X and y so the train/test positions
    # returned by train_test_split line up with ``y.loc[...]`` even when ``df``
    # is a sample whose index is not 0..n-1 (e.g. rows selected from the full
    # dataset). Resetting only y silently misaligned the labels in that case.
    y = df[target_col]

    # Split the RAW rows first so evaluate_on_test_set() can recover the exact
    # same held-out rows. idx_* are positions into X/y (both indexed 0..n-1).
    idx_train, idx_test = train_test_split(
        X.index, test_size=0.2, random_state=42, stratify=y
    )

    # Fit the large-amount threshold on the training rows only and persist it
    # in the bundle, removing both the test leak and the train/serve skew.
    amount_col = resolve_column(X, "amount")
    threshold = (
        pd.to_numeric(X.loc[idx_train, amount_col], errors="coerce").median()
        if amount_col is not None
        else None
    )

    eng_df = engineer_features(
        pd.DataFrame(X).assign(**{target_col: y.values}),
        large_amount_threshold=threshold,
    )
    feature_cols = get_feature_columns(eng_df)

    X_train = eng_df.loc[idx_train, feature_cols]
    X_test = eng_df.loc[idx_test, feature_cols]
    y_train = y.loc[idx_train]
    y_test = y.loc[idx_test]

    bundle = train_model(X_train, y_train, model_type)
    bundle["large_amount_threshold"] = threshold
    bundle["test_index"] = idx_test
    eval_res = evaluate_model(bundle, X_test, y_test)
    print(eval_res["classification_report"])
    return save_model(bundle)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    path = build_and_save_model(model_type="random_forest")
    print(f"Model saved to: {path}")
