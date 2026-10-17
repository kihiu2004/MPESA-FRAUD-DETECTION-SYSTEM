import os
from pathlib import Path

import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

# PROJECT PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
# Saved gzip-compressed (~150 MB instead of ~490 MB) to fit limited disk space.
# pandas reads .gz files transparently with pd.read_csv().
RAW_DATA_FILE = RAW_DATA_DIR / "paysim.csv.gz"
# Earlier runs of this project cached the download under src/data/raw. Keep
# reading that copy when the canonical location is empty so an existing
# ~180 MB cache is never thrown away and re-downloaded.
LEGACY_RAW_DATA_FILE = PROJECT_ROOT / "src" / "data" / "raw" / "paysim.csv.gz"

# Kaggle dataset handle must be in "owner/dataset" format.
# PaySim dataset: https://www.kaggle.com/datasets/ealaxi/paysim1
DATASET_NAME = "ealaxi/paysim1"
DATASET_FILE = "PS_20174392719_1491204439457_log.csv"

# LOAD PAYSIM DATASET FROM KAGGLE
def load_paysim_data():
    print("loading paysim dataset....")

    # DOWNLOAD THE LATEST DATASET FROM KAGGLE
    # Note: Kaggle serves this file as a ZIP archive (with a .csv name),
    # so tell pandas to decompress it while reading.
    try:
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            DATASET_NAME,
            DATASET_FILE,
            pandas_kwargs={"compression": "zip"},
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download/load PaySim from Kaggle: {e}") from e

    print("DATASET LOADED SUCCESSFULLY")
    print(f"Dataset shape: {df.shape}")
    return df

# VALIDATE THE BASIC STRUCTURE OF THE PAYSIM DATASET
def validate_dataset(df):
    required_columns = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud"
    ]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    print("Dataset validation successful.")
    return True

# SAVE THE RAW PAYSIM DATASET TO THE PROJECTS RAW DATA DIRECTORY
def save_raw_data(df):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first, then atomically replace the target so a
    # crash mid-write can never leave a truncated cache file behind.
    # Path.with_suffix() would REPLACE the ".gz" suffix (giving
    # "paysim.csv.csv.gz.tmp"), so append the temp marker to the full name.
    tmp_file = RAW_DATA_FILE.parent / (RAW_DATA_FILE.name + ".tmp")
    try:
        df.to_csv(tmp_file, index=False, compression="gzip")
        os.replace(tmp_file, RAW_DATA_FILE)
    except Exception as e:
        tmp_file.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to save raw data to {RAW_DATA_FILE}: {e}") from e
    print(f"Raw dataset saved to: {RAW_DATA_FILE}")
    return RAW_DATA_FILE

# COMPLETE DATA LOADING PIPELINE
def load_and_save_data():
    # Use the locally saved raw file if it already exists, to avoid
    # re-downloading and re-decompressing on every run. The canonical
    # data/raw location is preferred, but the legacy src/data/raw cache is
    # accepted so older checkouts keep working without a fresh download.
    cached_file = None
    for candidate in (RAW_DATA_FILE, LEGACY_RAW_DATA_FILE):
        if candidate.exists():
            cached_file = candidate
            break

    if cached_file is not None:
        print(f"Loading cached raw data from: {cached_file}")
        try:
            df = pd.read_csv(cached_file)
            validate_dataset(df)
            return df
        except Exception as e:
            print(f"Cached file unreadable or invalid ({e}); re-downloading.")
            try:
                cached_file.unlink()
            except OSError:
                pass
    df = load_paysim_data()
    validate_dataset(df)
    save_raw_data(df)
    return df

if __name__ == "__main__":
    df = load_and_save_data()
    print("First 5 rows")
    print(df.head())

    print("columns")
    print(df.columns.tolist())

    print("Fraud distribution")
    print(df["isFraud"].value_counts())
