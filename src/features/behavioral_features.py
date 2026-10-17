import pandas as pd
import numpy as np


def count_within_window(steps, window=24):
    """How many of a customer's transactions fall within ``window`` steps
    (hours) of each transaction, including the current one.

    The input must be the customer's ``step`` values sorted ascending. Uses
    ``np.searchsorted`` so each group is handled in O(n log n) instead of a
    naive double loop.
    """
    arr = steps.to_numpy(dtype="float64")
    # First index whose step is at least current - window + 1, i.e. the first
    # transaction inside the trailing 24-hour window.
    lower = np.searchsorted(arr, arr - window + 1, side="left")
    return pd.Series(np.arange(len(arr)) - lower + 1, index=steps.index)


# MATCH A COLUMN NAME IGNORING CASE AND SURROUNDING WHITESPACE, SO THE FEATURES
# ARE ALSO CREATED FOR THE LOWERCASED COLUMNS PRODUCED BY
# data_cleaning.clean_transaction_data ("nameorig" INSTEAD OF "nameOrig").
# RETURNS None WHEN ABSENT.
def resolve_column(df, name):
    lookup = {str(column).strip().lower(): column for column in df.columns}
    return lookup.get(name.strip().lower())


# CREATE CUSTOMERS BEHAVIORAL FEATURES FOR FRAUD DETECTION
def create_behavioral_features(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be pandas DataFrame.")
    if df.empty:
        raise ValueError("The DataFrame is empty.")

    data = df.copy()

    # RESOLVE THE REQUIRED COLUMNS REGARDLESS OF THEIR CASE, THEN VALIDATE
    required_columns = {
        "nameOrig": resolve_column(data, "nameOrig"),
        "amount": resolve_column(data, "amount"),
        "type": resolve_column(data, "type"),
        "step": resolve_column(data, "step"),
    }
    missing_columns = [
        name for name, column in required_columns.items()
        if column is None
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    name_col = required_columns["nameOrig"]
    amount_col = required_columns["amount"]
    type_col = required_columns["type"]
    step_col = required_columns["step"]
    dest_col = resolve_column(data, "nameDest")

    # COERCE THE NUMERIC INPUTS ONCE, THE SAME WAY create_transaction_features
    # DOES. amount/step WERE USED RAW, SO A CSV THAT FAILED NUMERIC TYPE
    # INFERENCE (E.G. "1000" READ AS STRINGS) MADE cummax/cummin RAISE
    # "cummax is not supported for str dtype" AND THE expanding AGGREGATES RAISE
    # "operation 'sub' not supported for dtype 'str'". to_numeric IS A NO-OP
    # FOR REAL NUMERIC DATA.
    data[amount_col] = pd.to_numeric(data[amount_col], errors="coerce")
    data[step_col] = pd.to_numeric(data[step_col], errors="coerce")

    # REMEMBER THE CALLER'S INDEX/ROW ORDER; ALL WORK BELOW USES A FRESH
    # POSITIONAL RANGE INDEX SO NOTHING DEPENDS ON (OR MUTATES) THE INPUT INDEX.
    original_index = data.index
    data = data.reset_index(drop=True)

    # THE VELOCITY FEATURE HAS TO SEE EACH CUSTOMER'S TRANSACTIONS IN TIME
    # ORDER, SO SORT BY CUSTOMER AND STEP AND RESTORE THE ORIGINAL ORDER AT THE
    # END. THE SORT PERMUTATION IS KEPT AS A PLAIN POSITION ARRAY - NO TEMPORARY
    # MARKER COLUMN IS ADDED TO THE DATA, SO A USER COLUMN CAN NEVER BE
    # OVERWRITTEN AND NO NAME COLLISION IS POSSIBLE. WITHOUT THIS THE FUNCTION
    # USED TO RETURN A ROW-SHUFFLED FRAME, WHICH SILENTLY MISALIGNED VALUES AND
    # THE isFraud LABEL WHEN MERGED/CONCATENATED WITH OTHER FEATURES.
    sort_positions = (
        data[[name_col, step_col]]
        .sort_values([name_col, step_col], kind="stable")
        .index.to_numpy()
    )
    data = data.iloc[sort_positions].reset_index(drop=True)

    # ALL AGGREGATES BELOW ARE "AS OF NOW": THEY USE ONLY THE CURRENT AND PAST
    # TRANSACTIONS ("EXPANDING" WINDOW), NEVER LATER ONES. THE ORIGINAL
    # VERSIONS USED EACH CUSTOMER'S FULL HISTORY, WHICH LEAKED FUTURE
    # TRANSACTIONS INTO THE FEATURES (ESPECIALLY BAD FOR TIME-BASED SPLITS).
    # dropna=False KEEPS ROWS WITH A MISSING CUSTOMER KEY IN THE AGGREGATES.
    # WITH THE DEFAULT (dropna=True) pandas DROPPED THOSE ROWS, SO THE
    # "customer_avg_amount" EXPANDING MEAN CAME BACK SHORTER THAN THE FRAME AND
    # ASSIGNMENT RAISED "Length of values (N) does not match length of index".
    by_customer = data.groupby(name_col, sort=False, dropna=False)

    # TRANSACTION COUNT PER CUSTOMER (UP TO AND INCLUDING THIS ROW)
    data["customer_transaction_count"] = (
        by_customer.cumcount() + 1
    ).to_numpy()

    # AVERAGE TRANSACTION AMOUNT PER CUSTOMER (UP TO AND INCLUDING THIS ROW)
    data["customer_avg_amount"] = (
        by_customer[amount_col]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )

    # MAXIMUM TRANSACTION AMOUNT PER CUSTOMER (UP TO AND INCLUDING THIS ROW)
    data["customer_max_amount"] = by_customer[amount_col].cummax().to_numpy()

    # MINIMUM TRANSACTION AMOUNT PER CUSTOMER (UP TO AND INCLUDING THIS ROW)
    data["customer_min_amount"] = by_customer[amount_col].cummin().to_numpy()

    # TRANSACTION AMOUNT STANDARD DEVIATION (UP TO AND INCLUDING THIS ROW),
    # 0 FOR THE FIRST TRANSACTION OF A CUSTOMER WHERE std IS UNDEFINED
    data["customer_amount_std"] = (
        by_customer[amount_col]
        .expanding()
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
        .to_numpy()
    )

    # RATIO OF THE CURRENT AMOUNT TO CUSTOMER AVERAGE SO FAR
    data["amount_to_customer_avg_ratio"] = np.where(
        data["customer_avg_amount"] > 0,
        data[amount_col] / data["customer_avg_amount"],
        0,
    )

    # CUSTOMER TRANSACTION FREQUENCY BY TRANSACTION TYPE (UP TO THIS ROW)
    data["customer_type_count"] = (
        data.groupby([name_col, type_col], sort=False, dropna=False).cumcount() + 1
    ).to_numpy()

    # NUMBER OF UNIQUE TRANSACTION TYPES SEEN SO FAR. COUNTING THE FIRST TIME
    # EACH (CUSTOMER, TYPE) PAIR APPEARS AND RUNNING A CUMULATIVE SUM IS A
    # VECTORIZED "EXPANDING NUNIQUE".
    data["customer_unique_transaction_types"] = (
        (~data.duplicated(subset=[name_col, type_col]))
        .groupby(data[name_col], sort=False, dropna=False)
        .cumsum()
        .to_numpy()
    )

    # NUMBER OF UNIQUE RECIPIENTS SEEN SO FAR
    if dest_col is not None:
        data["customer_unique_recipients"] = (
            (~data.duplicated(subset=[name_col, dest_col]))
            .groupby(data[name_col], sort=False, dropna=False)
            .cumsum()
            .to_numpy()
        )

    # TRANSACTION HOUR. step IS 1 INDEXED, SO SUBTRACT 1 BEFORE WRAPPING TO A
    # 0-23 HOUR. THIS KEEPS transaction_hour / is_night_transaction CONSISTENT
    # WITH transaction_features.is_night, WHICH USES THE SAME CONVENTION.
    data["transaction_hour"] = (data[step_col] - 1) % 24

    # NIGHT TIME TRANSACTION INDICATOR
    data["is_night_transaction"] = (
        (data["transaction_hour"] < 6) | (data["transaction_hour"] >= 22)
    ).astype(int)

    # HIGH VALUE TRANSACTION INDICATOR (RELATIVE TO THE CUSTOMER'S AVERAGE SO FAR)
    data["is_high_value_transaction"] = (
        data[amount_col] > data["customer_avg_amount"] * 2
    ).astype(int)

    # AMOUNT Z_SCORE RELATIVE TO THE CUSTOMER'S BEHAVIOUR SO FAR
    data["amount_zscore"] = np.where(
        data["customer_amount_std"] > 0,
        (data[amount_col] - data["customer_avg_amount"]) / data["customer_amount_std"],
        0,
    )

    # TRANSACTION VELOCITY: HOW MANY TRANSACTIONS THE CUSTOMER MADE IN THE
    # LAST 24 STEPS (HOURS). THE DATA IS ALREADY SORTED BY CUSTOMER AND STEP
    # ABOVE, SO THE ROLLING COUNT IS CORRECT. A PLAIN
    # groupby(["nameOrig","step"]).size() ONLY COUNTS TRANSACTIONS IN THE
    # EXACT SAME HOUR AND TIMES OUT TO 1 FOR ANY CUSTOMER WITH ONE
    # TRANSACTION PER HOUR, WHICH MAKES THE VELOCITY FEATURE A PERMANENTLY
    # ZERO DEAD COLUMN.
    data["transactions_last_24_steps"] = (
        data.groupby(name_col, sort=False, dropna=False)[step_col].transform(count_within_window)
    )

    # RAPID TRANSACTION INDICATOR
    data["high_transaction_velocity"] = (
        data["transactions_last_24_steps"] >= 3
    ).astype(int)

    # RESTORE THE ORIGINAL ROW ORDER AND INDEX (np.argsort OF A PERMUTATION IS
    # ITS INVERSE) SO THE OUTPUT LINES UP ROW-FOR-ROW WITH THE INPUT.
    data = data.iloc[np.argsort(sort_positions)].reset_index(drop=True)
    data.index = original_index

    return data


if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.data.data_loader import load_and_save_data

    df = load_and_save_data()
    features = create_behavioral_features(df)

    print("Features shape:", features.shape)
    print("New columns:")
    print(sorted(set(features.columns) - set(df.columns)))
    print(features.head())
