import pandas as pd
from sklearn.preprocessing import StandardScaler

# PROCESS TRANSACTION DATA FOR FRAUD DETECTION
def process_data(
    df,
    target_column="isFraud",
    drop_columns=None,
    scale_features=False
):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be pandas DataFrame.")
    if df.empty:
        raise ValueError("The DataFrame is empty.")

    # RESOLVE THE TARGET COLUMN IGNORING CASE SO THE DEFAULT "isFraud" STILL
    # MATCHES THE LOWERCASED "isfraud" PRODUCED BY clean_transaction_data.
    resolved_target = next(
        (column for column in df.columns if str(column).lower() == str(target_column).lower()),
        None,
    )
    if resolved_target is None:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset")
    target_column = resolved_target

    # MAKE A COPY SO THE ORIGINAL DATAFRAME IS NOT MODIFIED
    data = df.copy()

    # REMOVE USER SPECIFIED COLUMNS
    if drop_columns:
        columns_to_drop = [
            column for column in drop_columns
            if column in data.columns
        ]
        data = data.drop(columns=columns_to_drop)

    # SEPARATE TARGET VARIABLE
    y = data[target_column]
    x = data.drop(columns = [target_column])

    # CONVERT CATEGORICAL VARIABLES TO NUMERICAL VALUES
    categorical_columns = x.select_dtypes(
        include=["object", "category"]
    ).columns

    if len(categorical_columns) > 0:
        x = pd.get_dummies(
            x,
            columns=categorical_columns,
            drop_first=True
        )

    # MAKE SURE ALL FEATURES ARE NUMERIC
    x = x.apply(pd.to_numeric, errors="coerce")

    # HANDLE MISSING VALUES
    x = x.fillna(0)

    # OPTIONAL FEATURES SCALING
    if scale_features:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        x = pd.DataFrame(
            x_scaled,
            columns=x.columns,
            index=x.index
        )

    return x, y


if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.data.data_loader import load_and_save_data

    df = load_and_save_data()
    x, y = process_data(
        df,
        drop_columns=["nameOrig", "nameDest", "isFlaggedFraud"],
        scale_features=True,
    )
    print("Features shape:", x.shape)
    print("Target distribution:")
    print(y.value_counts())
    print(x.head())
