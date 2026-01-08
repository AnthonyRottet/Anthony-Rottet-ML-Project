import pandas as pd
from pathlib import Path

# --- FILE PATHS ---
RAW_PATH = Path("../data/raw/heart_disease.csv")
PROCESSED_DIR = Path("../data/processed")
FULL_CLEAN_PATH = PROCESSED_DIR / "heart_disease_clean.csv"  # Version with all variables
MINI_CLEAN_PATH = PROCESSED_DIR / "heart_disease_clean_mini.csv"  # Version for finalized models

# Ensure the output directory exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not RAW_PATH.exists():
        print(f"Error: {RAW_PATH} not found.")
        return

    df = pd.read_csv(RAW_PATH)

    print("\n=== RAW DATA ===")
    print("Initial Shape:", df.shape)

    # 1. Standardize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # 2. Map heart disease variable to binary labels
    target_col = "heart_disease"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # Convert "Presence"/"Absence" to 1/0
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

    # Check for mapping errors
    if df[target_col].isna().any():
        print("Unique values in target column before failure:", df[target_col].unique())
        raise ValueError("Target mapping failed. Unrecognized values found in target column.")

    print("\n[INFO] Target successfully converted to binary labels.")

    # 3. SAVE FULL VERSION (13 Variables + Target)
    # This is done before any columns are dropped
    df.to_csv(FULL_CLEAN_PATH, index=False)
    print(f"\n[OK] Full dataset (13 variables) saved to: {FULL_CLEAN_PATH}")
    print("Shape:", df.shape)

    # 4. Pruning non-informative features based on Mutual Information (MI)
    cols_to_remove = ["age", "bp", "fbs_over_120"]
    print(f"\n[INFO] Pruning non-informative features: {cols_to_remove}")

    # Create the mini version
    df_mini = df.drop(columns=cols_to_remove)

    # 5. Missing values check
    print("\n=== MISSING VALUES CHECK (MINI VERSION) ===")
    missing = df_mini.isna().sum()
    if missing.sum() > 0:
        print(f"Warnings found:\n{missing[missing > 0]}")
    else:
        print("[OK] No missing values detected.")

    # 6. Export the mini dataset (Finalized features)
    df_mini.to_csv(MINI_CLEAN_PATH, index=False)

    print("\n=== SAVED PROCESSED DATA ===")
    print("Path:", MINI_CLEAN_PATH)
    print("Final Shape:", df_mini.shape)
    print("Final Variable Set:", [c for c in df_mini.columns if c != target_col])


if __name__ == "__main__":
    main()