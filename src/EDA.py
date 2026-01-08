import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
# Points to the newly created clean file with 13 variables
DATA_PATH = Path("../data/processed/heart_disease_clean.csv")
RESULTS_DIR = Path("../results/EDA")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1. Load Data
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Run your preprocessing script first.")
        return

    df = pd.read_csv(DATA_PATH)

    # List of variables to investigate (where MI was 0)
    zero_mi_vars = ["age", "bp", "fbs_over_120"]
    target_col = "heart_disease"

    sns.set_theme(style="whitegrid")

    # 2. Loop through variables and create plots
    for var in zero_mi_vars:
        if var not in df.columns:
            print(f"Skipping {var}: Column not found in dataset.")
            continue

        plt.figure(figsize=(10, 5))

        if var == "fbs_over_120":
            # Categorical Plot for FBS using the binary 0/1 target directly
            sns.countplot(data=df, x=var, hue=target_col, palette='viridis')
            plt.title(f"Distribution of Heart Disease (0/1) across {var.upper()}", fontsize=14)
            plt.xlabel("FBS > 120 mg/dl (0=No, 1=Yes)")
            plt.legend(title="Heart Disease", labels=["Absence (0)", "Presence (1)"])
        else:
            # KDE Plot for Continuous Variables (Age/BP)
            sns.kdeplot(data=df, x=var, hue=target_col, fill=True,
                        common_norm=False, palette='magma', alpha=0.5)
            plt.title(f"Distribution Overlap: {var.upper()} vs Heart Disease", fontsize=14)
            plt.xlabel(f"{var.capitalize()} Value")
            # Accessing the legend to update labels for clarity in the plot
            plt.gca().get_legend().set_title("Heart Disease")

        plt.ylabel("Density / Count")
        plt.tight_layout()

        # Save each plot
        file_name = f"eda_{var}.png"
        save_path = RESULTS_DIR / file_name
        plt.savefig(save_path, dpi=300)
        print(f"Generated plot for {var}: {save_path}")



if __name__ == "__main__":
    main()