import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.detection.fraud_model import (
    engineer_features,
    evaluate_model,
    get_feature_columns,
    load_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_RESULTS_PATH = PROJECT_ROOT / "models" / "evaluation_results.json"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_transactions.csv"
# Upper bound on the rows loaded for the legacy fallback below (models saved
# before test_index existed), so evaluation can never OOM on the 6.3M rows.
MAX_EVAL_ROWS = 200_000


def _read_rows_by_position(positions, chunk_size: int = 250_000) -> pd.DataFrame:
    """Return only the CSV rows whose 0-based position is in ``positions``.

    The file is streamed in chunks so scoring a model trained on a sample never
    has to load the full 6.3M-row dataset into memory.
    """
    wanted = np.asarray(sorted({int(p) for p in positions}), dtype=np.int64)
    parts = []
    offset = 0
    for chunk in pd.read_csv(PROCESSED_DATA_PATH, chunksize=chunk_size):
        chunk.index = np.arange(offset, offset + len(chunk))
        offset += len(chunk)
        mask = chunk.index.isin(wanted)
        if mask.any():
            parts.append(chunk[mask])
    if not parts:
        raise ValueError("None of the held-out test rows were found in the data")
    return pd.concat(parts)


def evaluate_on_test_set() -> Dict:
    bundle = load_model()

    header = pd.read_csv(PROCESSED_DATA_PATH, nrows=0)
    target_col = next((c for c in ("isfraud", "isFraud") if c in header.columns), None)
    if target_col is None:
        raise ValueError("Target column 'isfraud' not found in data")

    # Score only the rows the model was NOT trained on. Prefer the split stored
    # at training time; fall back to recreating the deterministic split for
    # models saved before that metadata existed.
    test_index = bundle.get("test_index")
    threshold = bundle.get("large_amount_threshold")

    if test_index is None:
        df = pd.read_csv(PROCESSED_DATA_PATH, nrows=MAX_EVAL_ROWS)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        _, test_index = train_test_split(
            X.index, test_size=0.2, random_state=42, stratify=y
        )
        if threshold is None:
            amount_col = "amount" if "amount" in df.columns else None
            threshold = df[amount_col].median() if amount_col is not None else None
        df_test = df.loc[test_index]
    else:
        df_test = _read_rows_by_position(test_index)

    X_test_raw = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    eng_df = engineer_features(
        pd.DataFrame(X_test_raw).assign(**{target_col: y_test.values}),
        large_amount_threshold=threshold,
    )
    feature_cols = get_feature_columns(eng_df)

    results = evaluate_model(bundle, eng_df[feature_cols], y_test)
    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in results.items()}, f, indent=2)
    print(f"Evaluated on {len(df_test)} held-out rows.")
    return results


if __name__ == "__main__":
    res = evaluate_on_test_set()
    print(res)
