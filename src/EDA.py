import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION ---
DATA_PATH = Path("../data/processed/heart_disease_clean.csv")
RESULTS_DIR = Path("../results/EDA")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    #Load Data
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Run your preprocessing script first.")
        return

    df = pd.read_csv(DATA_PATH)
    target_col = "heart_disease"
    sns.set_theme(style="white")
    sns.set_context("poster", font_scale=1.1)

    #correlation matrix
    print("Generating Correlation Matrix with bold numbers...")
    plt.figure(figsize=(22, 16))

    corr_matrix = df.corr()

    #Heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,  # Show the numbers
        fmt=".2f",  # 2 decimal places
        cmap='coolwarm',
        center=0,
        # annot_kws controls the numbers inside the boxes
        annot_kws={"size": 22, "weight": "bold"},
        linewidths=2,
        cbar_kws={"shrink": .8}
    )

    plt.title("FEATURE CORRELATION MATRIX", fontsize=45, fontweight='bold', pad=50)
    plt.xticks(fontsize=24, fontweight='bold', rotation=45, ha='right')
    plt.yticks(fontsize=24, fontweight='bold')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    #create the plot for the var with a MI of 0
    zero_mi_vars = ["age", "bp", "fbs_over_120"]

    for var in zero_mi_vars:
        if var not in df.columns:
            continue

        plt.figure(figsize=(16, 9))

        if var == "fbs_over_120":
            ax = sns.countplot(data=df, x=var, hue=target_col, palette='Set2')
            plt.title(f"DISTRIBUTION: {var.upper()}", fontsize=40, fontweight='bold', pad=40)
            plt.xlabel("FBS > 120 mg/dl (0=No, 1=Yes)", fontsize=30, fontweight='bold')

            # Larger legend text
            plt.legend(title="Heart Disease", labels=["Absence (0)", "Presence (1)"],
                       title_fontsize=24, fontsize=22, loc='upper right', frameon=True)
        else:
            ax = sns.kdeplot(data=df, x=var, hue=target_col, fill=True,
                             common_norm=False, palette='mako', alpha=0.4, linewidth=6)
            plt.title(f"OVERLAP ANALYSIS: {var.upper()}", fontsize=40, fontweight='bold', pad=40)
            plt.xlabel(f"{var.capitalize()} Value", fontsize=30, fontweight='bold')

            legend = ax.get_legend()
            if legend:
                plt.setp(legend.get_title(), fontsize=24, fontweight='bold')
                plt.setp(legend.get_texts(), fontsize=22)

        plt.ylabel("Density / Count", fontsize=30, fontweight='bold')
        plt.xticks(fontsize=26, fontweight='bold')
        plt.yticks(fontsize=26, fontweight='bold')

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"eda_{var}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated bold plot for {var}")


if __name__ == "__main__":
    main()