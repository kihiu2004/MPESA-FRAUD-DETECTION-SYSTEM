import numpy as np
import pandas as pd

# HUMAN READABLE REASON PRODUCED FOR EACH RULE, USED TO EXPLAIN A FLAG
RULE_REASONS = {
    "rule_large_amount": "Large transaction amount",
    "rule_balance_depletion": "Substantial origin balance depletion",
    "rule_balance_mismatch": "Amount exceeds recorded origin balance",
    "rule_sensitive_transaction_type": "Transfer or cash_out requires review",
}


# MATCH A COLUMN NAME IGNORING CASE AND SURROUNDING WHITESPACE, SO THE RULES
# ALSO WORK ON THE LOWERCASED COLUMNS PRODUCED BY
# data_cleaning.clean_transaction_data ("oldbalanceorg" INSTEAD OF
# "oldbalanceOrg"). RETURNS None WHEN ABSENT.
def resolve_column(df, name):
    lookup = {str(column).strip().lower(): column for column in df.columns}
    return lookup.get(str(name).strip().lower())


# TRANSLATE THE VERIFIED RULE COLUMNS OF A SINGLE ROW INTO READABLE REASONS
def get_rule_reasons(row):
    return [
        reason
        for column, reason in RULE_REASONS.items()
        if bool(row[column])
    ]


# DETECT POTENTIALLY FRAUDULENT TRANSACTIONS USING PREDEFINED RULES
def apply_fraud_rules(
    df,
    amount_column="amount",
    transaction_type_column="type",
    old_balance_column="oldbalanceOrg",
    new_balance_column="newbalanceOrig",
    amount_threshold=200000,
    balance_drop_threshold=0.90,
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be pandas DataFrame.")
    if df.empty:
        raise ValueError("The DataFrame is empty.")

    # RESOLVE THE COLUMN NAMES REGARDLESS OF CASE BEFORE VALIDATING THEM
    requested_columns = {
        "amount_column": amount_column,
        "transaction_type_column": transaction_type_column,
        "old_balance_column": old_balance_column,
        "new_balance_column": new_balance_column,
    }
    missing_columns = [
        name for name, column in requested_columns.items()
        if resolve_column(df, column) is None
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            "Expected amount, type, oldbalanceOrg and newbalanceOrig."
        )

    amount_column = resolve_column(df, amount_column)
    transaction_type_column = resolve_column(df, transaction_type_column)
    old_balance_column = resolve_column(df, old_balance_column)
    new_balance_column = resolve_column(df, new_balance_column)

    if amount_threshold < 0:
        raise ValueError("amount_threshold cannot be negative.")

    if not 0 < balance_drop_threshold <= 1:
        raise ValueError(
            "balance_drop_threshold must be between 0 and 1."
        )

    result = df.copy()

    # CONVERT NUMERIC FIELDS SAFELY
    amount = pd.to_numeric(
        result[amount_column],
        errors="coerce",
    )

    old_balance = pd.to_numeric(
        result[old_balance_column],
        errors="coerce",
    )

    new_balance = pd.to_numeric(
        result[new_balance_column],
        errors="coerce",
    )

    # UNUSUALLY LARGE TRANSACTIONS
    result["rule_large_amount"] = amount > amount_threshold

    # ORIGIN BALANCE DROPS SUBSTANTIALLY
    balance_drop = old_balance - new_balance

    balance_drop_ratio = (
        balance_drop / old_balance.where(old_balance > 0)
    )

    result["rule_balance_depletion"] = (
        (old_balance > 0)
        & (balance_drop_ratio > balance_drop_threshold)
    )

    # TRANSACTION AMOUNT EXCEEDS AVAILABLE ORIGIN BALANCE
    result["rule_balance_mismatch"] = (
        amount.notna()
        & old_balance.notna()
        & (amount > old_balance)
    )

    # TRANSACTION TYPE IS TRANSFER OR CASH_OUT
    transaction_type = (
        result[transaction_type_column]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    result["rule_sensitive_transaction_type"] = transaction_type.isin(
        ["TRANSFER", "CASH_OUT"]
    )

    # COUNT THE RULES TRIGGERED BY EACH TRANSACTION. rule_columns COMES FROM
    # RULE_REASONS SO THE FLAGS, THE COUNT AND THE REASONS CAN NEVER DRIFT
    # APART (THE ORIGINAL FILE HAD A TYPO'D COLUMN NAME HERE).
    rule_columns = list(RULE_REASONS)
    result["rules_triggered"] = (
        result[rule_columns].sum(axis=1).astype(int)
    )

    # FLAGGED TRANSACTIONS TRIGGERED AT LEAST ONE RULE
    result["rule_based_flag"] = result["rules_triggered"] > 0

    # EXPLAIN WHY EACH TRANSACTION WAS FLAGGED. A LIST COMPREHENSION OVER A
    # BOOLEAN MATRIX IS USED INSTEAD OF DataFrame.apply(axis=1), WHICH IS
    # ORDERS OF MAGNITUDE SLOWER ON THE FULL 6M+ ROW PAYSIM DATASET.
    triggered = result[rule_columns].to_numpy(dtype=bool)
    reason_labels = np.array(list(RULE_REASONS.values()), dtype=object)
    result["rule_reasons"] = [
        list(reason_labels[row]) for row in triggered
    ]

    return result


if __name__ == "__main__":
    samples = pd.DataFrame(
        {
            "type": ["TRANSFER", "CASH_OUT", "PAYMENT", "TRANSFER"],
            "amount": [250000.0, 50.0, 10.0, 5000.0],
            "oldbalanceOrg": [300000.0, 40.0, 100.0, 1000.0],
            "newbalanceOrig": [50000.0, 0.0, 90.0, 200.0],
        }
    )

    output = apply_fraud_rules(samples)
    print("Synthetic smoke test:")
    print(
        output[
            ["type", "amount", "rules_triggered", "rule_based_flag", "rule_reasons"]
        ].to_string()
    )
    print("get_rule_reasons(sample 0):", get_rule_reasons(output.iloc[0]))
