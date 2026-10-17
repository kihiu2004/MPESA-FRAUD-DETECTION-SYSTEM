import gc
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_cleaning import clean_transaction_data_streamed
from src.data.data_loader import (
    LEGACY_RAW_DATA_FILE,
    RAW_DATA_FILE,
    load_and_save_data,
)
from src.detection.anomaly_detection import save_anomaly_model, train_anomaly_detector
from src.detection.fraud_model import build_and_save_model, sample_processed_data
from src.evaluation.model_evaluation import evaluate_on_test_set
from src.features.behavioral_features import create_behavioral_features
from src.features.transaction_features import create_transaction_features
from src.utils.risk_scoring import combine_risk_scores

CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_transactions.csv"
FEATURED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "featured_transactions.csv"

# THE FULL PAYSIM DATASET (6.3M+ ROWS) DOES NOT FIT IN RAM ON AN 8 GB MACHINE
# ONCE THE TRANSACTION + BEHAVIOURAL FEATURES ARE BUILT: EVERY ENGINEERING STEP
# MAKES ANOTHER FULL COPY (~2.5 GB PER 49 FLOAT COLUMNS). THE PIPELINE
# THEREFORE NEVER MATERIALISES THE WHOLE ENGINEERED FRAME:
#   * TRAINING USES THE SAMPLING PATH ALREADY BUILT INTO fraud_model
#     (DEFAULT_MAX_TRAINING_ROWS) INSTEAD OF BEING HANDED THE FULL DataFrame.
#   * THE EXPLORATORY FEATURE EXPORT IS CAPPED TO A BOUNDED SLICE.
FEATURE_EXPORT_ROWS = 200_000
ANOMALY_TRAIN_ROWS = 100_000


def _has_cleaned_data() -> bool:
    return CLEANED_DATA_PATH.exists() and CLEANED_DATA_PATH.stat().st_size > 0


def _find_raw_cache() -> Path | None:
    for candidate in (RAW_DATA_FILE, LEGACY_RAW_DATA_FILE):
        if candidate.exists():
            return candidate
    return None


# MAKE SURE data/processed/cleaned_transactions.csv EXISTS. REUSE THE FILE WHEN
# IT IS ALREADY THERE SO A NORMAL RUN NEVER HAS TO HOLD THE FULL 6.3M ROW RAW
# FRAME IN MEMORY. PASS --refresh (OR DELETE THE CSV) TO REBUILD FROM RAW.
# REBUILDING STREAMS THE RAW FILE IN CHUNKS, SO EVEN A FIRST RUN / --refresh
# STAYS MEMORY-BOUNDED INSTEAD OF LOADING ALL 6.3M ROWS AT ONCE.
def prepare_cleaned_data(force_refresh: bool = False) -> None:
    if _has_cleaned_data() and not force_refresh:
        print(f"Using existing cleaned dataset: {CLEANED_DATA_PATH}")
        print("(pass --refresh to rebuild it from the raw PaySim download)")
        return

    raw_path = _find_raw_cache()
    if raw_path is None:
        # NO LOCAL CACHE: DOWNLOAD ONCE (kagglehub RETURNS A DataFrame), PERSIST
        # IT AS THE RAW CACHE, THEN FREE IT AND STREAM FROM DISK BELOW.
        print("No cached raw data found; downloading from Kaggle...")
        df = load_and_save_data()
        del df
        gc.collect()
        raw_path = _find_raw_cache()
        if raw_path is None:
            raise RuntimeError("Raw PaySim data is unavailable; cannot rebuild.")

    print(f"Cleaning raw data (streamed) from: {raw_path}")
    clean_transaction_data_streamed(raw_path, CLEANED_DATA_PATH)
    print(f"Cleaned data saved: {CLEANED_DATA_PATH}")


# WRITE featured_transactions.csv FROM A MEMORY-BOUNDED SLICE. THE BEHAVIOURAL
# FEATURES AGGREGATE A CUSTOMER'S ENTIRE HISTORY, SO THEY CANNOT BE STREAMED IN
# INDEPENDENT CHUNKS WITHOUT CHANGING THEIR MEANING; THIS FILE IS A DIAGNOSTIC
# ARTIFACT ONLY (TRAINING ENGINEERS ITS OWN FEATURES ON ITS OWN SAMPLE).
def export_feature_sample() -> None:
    print("\n2. Engineering features (bounded sample)...")
    sample = pd.read_csv(CLEANED_DATA_PATH, nrows=FEATURE_EXPORT_ROWS)
    threshold = sample["amount"].median()
    sample = create_transaction_features(sample, large_amount_threshold=threshold)
    sample = create_behavioral_features(sample)
    sample.to_csv(FEATURED_DATA_PATH, index=False)
    print(f"Featured data saved: {sample.shape} (first {FEATURE_EXPORT_ROWS} rows)")
    del sample
    gc.collect()


def main() -> None:
    print("M-Pesa Fraud Detection System")
    print("=" * 40)

    force_refresh = "--refresh" in sys.argv[1:]

    print("\n1. Preparing cleaned data...")
    prepare_cleaned_data(force_refresh=force_refresh)

    export_feature_sample()

    print("\n3. Training fraud detection model (memory-bounded sample)...")
    model_path = build_and_save_model()
    print(f"Model saved: {model_path}")

    print("\n4. Training anomaly detection model (memory-bounded sample)...")
    # sample_processed_data STREAMS THE CSV AND KEEPS ONLY A RANDOM SLICE, SO
    # THIS STAYS MEMORY-BOUNDED *AND* UNBIASED (A HEAD/first-N ROWS SLICE WOULD
    # ONLY EVER SEE THE FIRST FEW HOURS OF THE SIMULATION).
    anomaly_df = sample_processed_data(max_rows=ANOMALY_TRAIN_ROWS, random_state=42)
    anom_path = save_anomaly_model(train_anomaly_detector(anomaly_df))
    print(f"Anomaly model saved: {anom_path}")
    del anomaly_df
    gc.collect()

    print("\n5. Evaluating model...")
    evaluate_on_test_set()
    print("Evaluation completed")

    print("\n6. Testing risk scoring...")
    test_df = pd.read_csv(CLEANED_DATA_PATH, nrows=5)
    risk_res = combine_risk_scores(test_df)
    print(risk_res[["amount", "type", "risk_score", "risk_level"]])

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
