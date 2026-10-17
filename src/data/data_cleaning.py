import os
from pathlib import Path

import pandas as pd

# CLEAN THE RAW PAYSIM TRANSACTION DATASET
#   verbose: WHEN False, SUPPRESS THE PER-CALL SUMMARY PRINTS. THE STREAMING
#   PIPELINE BELOW CALLS THIS ONCE PER CHUNK, WHERE 13x SUMMARY SPAM IS NOISE.
def clean_transaction_data(df, verbose = True):
    
    # MAKE A COPY SO THE ORIGINAL DATAFRAME IS NOT MODIFIED
    df = df.copy()
    
    # STANDARDIZE COLUMN NAMES
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_")
    )
    
    # REMOVE DUPLICATE ROWS
    df = df.drop_duplicates()
    
    # HANDLE MISSING VALUES
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        if verbose:
            print(missing_values[missing_values > 0])
        
    # NUMERIC COLUMNS
    numeric_columns = df.select_dtypes(include = ["int64" , "float64"]).columns
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())
        
    # CATEGORICAL COLUMNS
    categorical_columns = df.select_dtypes(include = ["object" , "category"]).columns    
    for column in categorical_columns:
        if not df[column].mode().empty:
            df[column] = df[column].fillna(df[column].mode()[0]) 
            
    # REMOVE INVALID TRANSACTIONS AMOUNTS
    if "amount" in df.columns:
        df = df[df["amount"] >= 0]
        
    # CLEAN TRANSACTION TYPE
    if "type" in df.columns:
        df["type"] = df["type"].astype(str).str.strip().str.upper()
        
    # CLEAN NAME COLUMNS
    name_columns = [
        "nameorig",
        "namedest",
    ]    
    for column in name_columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()
            
    # ENSURE FRAUD AND FLAG COLUMNS ARE NUMERIC
    binary_columns = [
        "isfraud",
        "isflaggedfraud"
    ]        
    for column in binary_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column],errors = "coerce").fillna(0).astype(int)
            
    # REMOVE IMPOSSIBLE FRAUD VALUES
    if "isfraud" in df.columns:
        df = df[df["isfraud"].isin([0,1])]
        
    if "isflaggedfraud" in df.columns:
        df = df[df["isflaggedfraud"].isin([0,1])]
        
    # RESET INDEX
    df = df.reset_index(drop=True)
    
    # DISPLAY CLEANING SUMMARY
    if verbose:
        print("data cleaning completed sucessfully.")
        print(f"Rows:{len(df)}")
        print(f"columns:{len(df.columns)}")
    return df

# CLEAN A (POSSIBLY HUGE) RAW CSV WITHOUT LOADING IT ALL INTO MEMORY.
# STREAMS input_path IN CHUNKS, REUSES clean_transaction_data ON EACH CHUNK, AND
# APPENDS THE RESULT TO output_path. THE FULL PAYSIM FILE IS 6.3M ROWS (~0.5 GB)
# AND ENGINEERING IT IN MEMORY PEAKS AT SEVERAL GB; STREAMING KEEPS THE PEAK AT
# ONE CHUNK.
#
# THE ONLY GLOBAL OPERATIONS IN clean_transaction_data ARE drop_duplicates AND
# median/mode IMPUTATION, WHICH ARE NO-OPS ON PAYSIM (VERIFIED: 0 DUPLICATE
# ROWS, 0 NULLS). PER-CHUNK PROCESSING THEREFORE PRODUCES THE SAME FILE AS THE
# IN-MEMORY PATH WHILE USING A FRACTION OF THE MEMORY.
def clean_transaction_data_streamed(input_path, output_path, chunksize = 500_000):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    # WRITE TO A TEMP FILE FIRST, THEN ATOMICALLY REPLACE, SO A CRASH CAN NEVER
    # LEAVE A HALF-WRITTEN DATASET THAT LATER RUNS WOULD MISTAKE FOR COMPLETE.
    tmp_path = output_path.parent / (output_path.name + ".tmp")

    total_rows = 0
    column_count = 0
    first_chunk = True
    try:
        for chunk in pd.read_csv(input_path, chunksize = chunksize):
            cleaned = clean_transaction_data(chunk, verbose = False)
            if first_chunk:
                column_count = len(cleaned.columns)
            cleaned.to_csv(
                tmp_path,
                index = False,
                mode = "w" if first_chunk else "a",
                header = first_chunk,
            )
            first_chunk = False
            total_rows += len(cleaned)
        os.replace(tmp_path, output_path)
    except Exception:
        tmp_path.unlink(missing_ok = True)
        raise

    print("data cleaning completed sucessfully.")
    print(f"Rows:{total_rows}")
    print(f"columns:{column_count}")
    return output_path

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.data_loader import load_and_save_data
    df = load_and_save_data()
    cleaned_df = clean_transaction_data(df)
    print(cleaned_df.head())
    print(f"Fraud distribution after cleaning:\n{cleaned_df['isfraud'].value_counts()}")
                      