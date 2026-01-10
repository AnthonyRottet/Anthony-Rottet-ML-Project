import pandas as pd
from pathlib import Path

# --- FILE PATHS ---
RAW_PATH = Path("../data/raw/heart_disease.csv")
PROCESSED_DIR = Path("../data/processed")
FULL_CLEAN_PATH = PROCESSED_DIR / "heart_disease_clean.csv"
MINI_CLEAN_PATH = PROCESSED_DIR / "heart_disease_clean_mini.csv"

# Ensure the output directory exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not RAW_PATH.exists():
        print(f"Error: {RAW_PATH} not found.")
        return

    df = pd.read_csv(RAW_PATH)

    print("\n=== RAW DATA ===")
    print("Initial Shape:", df.shape)

    # standardize
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # map heart disease into binary
    target_col = "heart_disease"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # convert to 1/0
    df[target_col] = (
        df[target_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "presence": 1,
            "absence": 0
        })
    )

    # error handler
    if df[target_col].isna().any():
        print("Unique values in target column before failure:", df[target_col].unique())
        raise ValueError("Target mapping failed. Unrecognized values found in target column.")

    print("\n[INFO] Target successfully converted to binary labels.")

    #Save full version
    df.to_csv(FULL_CLEAN_PATH, index=False)
    print(f"\n[OK] Full dataset (13 variables) saved to: {FULL_CLEAN_PATH}")
    print("Shape:", df.shape)

    # remove the column with MI of 0
    cols_to_remove = ["age", "bp", "fbs_over_120"]
    print(f"\n[INFO] Pruning non-informative features: {cols_to_remove}")
    df_mini = df.drop(columns=cols_to_remove)

    # Check for missing values
    print("\n=== MISSING VALUES CHECK (MINI VERSION) ===")
    missing = df_mini.isna().sum()
    if missing.sum() > 0:
        print(f"Warnings found:\n{missing[missing > 0]}")
    else:
        print("[OK] No missing values detected.")

    # Export
    df_mini.to_csv(MINI_CLEAN_PATH, index=False)

    print("\n=== SAVED PROCESSED DATA ===")
    print("Path:", MINI_CLEAN_PATH)
    print("Final Shape:", df_mini.shape)
    print("Final Variable Set:", [c for c in df_mini.columns if c != target_col])


if __name__ == "__main__":
    main()