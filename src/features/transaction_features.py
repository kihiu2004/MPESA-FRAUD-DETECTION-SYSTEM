import warnings

import numpy as np
import pandas as pd

# PAYSIM TRANSACTION TYPE SEMANTICS USED BY THE BALANCE CONSISTENCY CHECKS
# CASH_IN IS THE ONLY TYPE THAT CREDITS THE ORIGIN ACCOUNT
CREDIT_ORIGIN_TYPES = {"CASH_IN"}
# PAYMENT SENDS MONEY TO A MERCHANT BALANCE THAT IS NEVER TRACKED, SO THE
# DESTINATION BALANCE OF A PAYMENT NEVER CHANGES
UNCHANGED_DEST_TYPES = {"PAYMENT"}
# CASH_IN TAKES THE MONEY FROM THE DESTINATION ACCOUNT, THE OTHER TYPES
# (TRANSFER, CASH_OUT, DEBIT) CREDIT THE DESTINATION ACCOUNT
DEBIT_DEST_TYPES = {"CASH_IN"}


# MATCH A COLUMN NAME IGNORING CASE SO THE FEATURES ARE ALSO CREATED FOR THE
# LOWERCASED COLUMNS PRODUCED BY data_cleaning.clean_transaction_data
# ("oldbalanceorg" INSTEAD OF "oldbalanceOrg"). RETURNS None WHEN ABSENT.
def resolve_column(df, name):
    lookup = {str(column).strip().lower(): column for column in df.columns}
    return lookup.get(name.lower())


# CREATE TRANSACTION FEATURES FOR FRAUD DETECTION
def create_transaction_features(df, large_amount_threshold=None):

    # VALIDATE INPUT THE SAME WAY process_data() DOES
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be pandas DataFrame.")
    if df.empty:
        raise ValueError("The DataFrame is empty.")

    # MAKE A COPY SO THAT THE ORIGINAL PAYSIM DATASET IS NOT MODIFIED
    data = df.copy()

    # RESOLVE COLUMN NAMES CASE INSENSITIVELY (SEE resolve_column ABOVE) SO
    # THE FEATURES ARE CREATED FOR RAW AND LOWERCASED (CLEANED) INPUT ALIKE
    amount_col = resolve_column(data, "amount")
    type_col = resolve_column(data, "type")
    step_col = resolve_column(data, "step")
    origin_old_col = resolve_column(data, "oldbalanceOrg")
    origin_new_col = resolve_column(data, "newbalanceOrig")
    dest_old_col = resolve_column(data, "oldbalanceDest")
    dest_new_col = resolve_column(data, "newbalanceDest")

    # NORMALISE THE TRANSACTION TYPE VALUES (TRIM + UPPERCASE) BEFORE THEY ARE
    # MATCHED. THE CHECKS BELOW COMPARE AGAINST EXACT UPPERCASE CONSTANTS
    # ("TRANSFER"), SO INPUT LIKE "transfer" SILENTLY PRODUCED ALL-ZERO TYPE
    # FLAGS AND WRONG EXPECTED BALANCES. data_cleaning AND rule_based BOTH
    # UPPERCASE THE TYPE, SO THIS KEEPS THE FEATURES CONSISTENT WITH THEM.
    if type_col:
        data[type_col] = (
            data[type_col]
            .astype("string")
            .str.strip()
            .str.upper()
            # A MISSING TYPE MUST STAY "NOT MATCHED": KEEPING pandas.NA WOULD
            # PROPAGATE INTO THE BOOLEAN FLAGS AND MAKE .astype(int) RAISE
            # "int() argument must be ... not 'NAType'".
            .fillna("")
        )

    # COERCE EVERY NUMERIC SOURCE COLUMN UP FRONT. amount ALWAYS WAS, BUT THE
    # BALANCE COLUMNS WERE USED RAW, SO OBJECT/STRING INPUT (E.G. A CSV THAT
    # FAILED TYPE INFERENCE) CRASHED pandas 3 WITH "operation 'sub' not
    # supported for dtype 'str'". to_numeric IS A NO-OP FOR REAL NUMERIC DATA.
    amount = None
    step = None
    origin_old = None
    origin_new = None
    dest_old = None
    dest_new = None

    if amount_col:
        amount = pd.to_numeric(data[amount_col], errors="coerce")
    if step_col:
        step = pd.to_numeric(data[step_col], errors="coerce")
    if origin_old_col:
        origin_old = pd.to_numeric(data[origin_old_col], errors="coerce")
    if origin_new_col:
        origin_new = pd.to_numeric(data[origin_new_col], errors="coerce")
    if dest_old_col:
        dest_old = pd.to_numeric(data[dest_old_col], errors="coerce")
    if dest_new_col:
        dest_new = pd.to_numeric(data[dest_new_col], errors="coerce")

    # TRACK EVERY COLUMN WE CREATE SO ONLY THOSE ARE CLEANED AT THE END
    created = []

    def add_feature(name, values):
        data[name] = values
        created.append(name)

    # TRANSACTION AMOUNT FEATURES
    if amount is not None:
        # clip(lower=0) KEEPS log1p DOMAIN VALID EVEN IF BAD ROWS SLIP THROUGH
        add_feature("amount_log", np.log1p(amount.clip(lower=0)))
        add_feature("amount_is_zero", (amount == 0).astype(int))
        # AFTER THE FINAL fillna(0) A MISSING AMOUNT LOOKS LIKE A REAL ZERO
        # (amount_log = 0) WHILE amount_is_zero SAYS OTHERWISE. THIS FLAG
        # TELLS THE MODEL WHICH OF THE TWO ROWS IT IS LOOKING AT.
        add_feature("amount_missing", amount.isna().astype(int))

    # BALANCE DIFFERENCE FEATURES
    if origin_old is not None and origin_new is not None:
        add_feature(
            "origin_balance_change",
            origin_old - origin_new,
        )

    if dest_old is not None and dest_new is not None:
        add_feature(
            "destination_balance_change",
            dest_new - dest_old,
        )

    # BALANCE CONSISTENCY FEATURES
    # THE EXPECTED BALANCE UPDATE DEPENDS ON THE DIRECTION OF THE TRANSACTION:
    # CASH_IN CREDITS THE ORIGIN, PAYMENT NEVER CHANGES THE DESTINATION AND
    # CASH_IN DRAWS FROM IT. WITH A FLAT "ORIGIN DEBITS / DESTINATION CREDITS"
    # RULE EVERY CASH_IN REPORTS 2 * AMOUNT AS ORIGIN ERROR AND EVERY PAYMENT
    # REPORTS AMOUNT AS DESTINATION ERROR, WHICH FLAGS ~ALL PAYMENTS AS
    # INCONSISTENT. BALANCES ARE FLOORED AT 0 IN THE SIMULATOR.
    # MISSING INPUTS LEAVE THE ERROR AS NaN, AND THE *_missing FLAG KEEPS
    # "NO DATA" FROM LOOKING LIKE "PERFECTLY CONSISTENT" AFTER THE FILL
    if amount is not None and origin_old is not None and origin_new is not None:
        if type_col:
            expected_origin = np.where(
                data[type_col].isin(CREDIT_ORIGIN_TYPES),
                origin_old + amount,
                np.maximum(0, origin_old - amount),
            )
        else:
            # NO TRANSACTION TYPE COLUMN: FALL BACK TO THE DEBIT ASSUMPTION
            expected_origin = np.maximum(0, origin_old - amount)
        origin_error = (
            pd.Series(expected_origin, index=data.index) - origin_new
        ).abs()
        add_feature("origin_balance_error", origin_error)
        add_feature("origin_balance_missing", origin_error.isna().astype(int))

    if amount is not None and dest_old is not None and dest_new is not None:
        if type_col:
            tx_type = data[type_col]
            expected_dest = np.where(
                tx_type.isin(UNCHANGED_DEST_TYPES),
                dest_old,
                np.where(
                    tx_type.isin(DEBIT_DEST_TYPES),
                    np.maximum(0, dest_old - amount),
                    dest_old + amount,
                ),
            )
        else:
            # NO TRANSACTION TYPE COLUMN: FALL BACK TO THE CREDIT ASSUMPTION
            expected_dest = dest_old + amount
        destination_error = (
            pd.Series(expected_dest, index=data.index) - dest_new
        ).abs()
        add_feature("destination_balance_error", destination_error)
        add_feature("destination_balance_missing", destination_error.isna().astype(int))
        add_feature(
            "destination_balance_inconsistent",
            (destination_error > 1).astype(int),
        )

    # ACCOUNT BALANCE RATIO FEATURES
    if amount is not None and origin_old is not None:
        # clip(lower=0) GUARANTEES A DENOMINATOR OF AT LEAST 1
        add_feature(
            "amount_to_origin_balance_ratio",
            amount / (origin_old.clip(lower=0) + 1),
        )

    # TRANSACTION TYPE FEATURES
    if type_col:
        add_feature("is_transfer", (data[type_col] == "TRANSFER").astype(int))
        add_feature("is_cash_out", (data[type_col] == "CASH_OUT").astype(int))
        add_feature("is_cash_in", (data[type_col] == "CASH_IN").astype(int))
        add_feature("is_payment", (data[type_col] == "PAYMENT").astype(int))
        add_feature("is_debit", (data[type_col] == "DEBIT").astype(int))

    # TRANSACTION TIME FEATURES
    # step IS 1 INDEXED, SO SUBTRACT 1 BEFORE WRAPPING TO A 0-23 HOUR
    if step is not None:
        hour = (step - 1) % 24
        day = (step - 1) // 24

        add_feature("hour", hour)
        add_feature("day", day)
        add_feature("is_night", ((hour < 6) | (hour >= 22)).astype(int))
        add_feature(
            "is_business_hours",
            ((hour >= 8) & (hour <= 17)).astype(int),
        )
        # fillna(0) TURNS A MISSING step INTO hour = 0 (MIDNIGHT, i.e. NIGHT)
        # WHILE THE BOOLEAN FLAGS STAY 0 BECAUSE NaN COMPARISONS ARE FALSE.
        # THIS FLAG MARKS THOSE ROWS SO THE CONTRADICTION IS EXPLICIT.
        add_feature("hour_missing", hour.isna().astype(int))

    # LARGE TRANSACTION INDICATOR
    if amount is not None:
        if large_amount_threshold is None:
            warnings.warn(
                "large_amount_threshold was not provided, falling back to the median "
                "of the supplied rows. Pass a threshold fitted on training data only "
                "to avoid leaking test data into this feature.",
                UserWarning,
                stacklevel=2,
            )
            large_amount_threshold = amount.median()

        # A NaN THRESHOLD OR AN EMPTY amount COLUMN WOULD MAKE
        # (amount > threshold) FALSE EVERYWHERE AND LEAVE A SILENTLY DEAD,
        # CONSTANT ZERO FEATURE, SO SKIP IT LOUDLY INSTEAD
        if pd.isna(large_amount_threshold) or not amount.notna().any():
            warnings.warn(
                "Skipping is_large_transaction: the threshold is NaN or the "
                "amount column has no valid values.",
                UserWarning,
                stacklevel=2,
            )
        else:
            add_feature(
                "is_large_transaction",
                (amount > large_amount_threshold).astype(int),
            )

    # ZERO BALANCE INDICATORS
    if origin_old is not None:
        add_feature(
            "origin_balance_zero",
            (origin_old == 0).astype(int),
        )

    # CLEAN ONLY THE DERIVED COLUMNS - RAW INPUT AND THE TARGET (isFraud)
    # ARE LEFT UNTOUCHED SO LABELS AND SOURCE VALUES ARE NEVER OVERWRITTEN.
    # COLUMNS ARE HANDLED ONE AT A TIME INSTEAD OF COPYING THE WHOLE BLOCK
    # SO PEAK MEMORY STAYS LOW ON THE FULL 6M+ ROW PAYSIM DATASET.
    # INTEGER COLUMNS CANNOT HOLD inf/NaN, SO THEY ARE SKIPPED ENTIRELY.
    for column in created:
        series = data[column]
        if not pd.api.types.is_float_dtype(series):
            continue
        data[column] = series.replace([np.inf, -np.inf], np.nan).fillna(0)

    return data


if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.data.data_loader import load_and_save_data

    df = load_and_save_data()
    features = create_transaction_features(df, large_amount_threshold=100000)

    print("Features shape:", features.shape)
    print("New columns:")
    print(sorted(set(features.columns) - set(df.columns)))
    print(features.head())
