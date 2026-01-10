import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif

# --- CONFIGURATION ---
SEED = 42
DATA_PATH = Path("../data/processed/heart_disease_clean.csv")
RESULTS_DIR = Path("../results/Mutual Information")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    #Load Data
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    target_col = "heart_disease"

    #features
    numeric_cols = ["age", "bp", "cholesterol", "max_hr", "st_depression"]
    categorical_cols = [
        "sex", "chest_pain_type", "fbs_over_120", "ekg_results",
        "exercise_angina", "slope_of_st", "number_of_vessels_fluro", "thallium"
    ]

    X = df[numeric_cols + categorical_cols]
    y = df[target_col].astype(int)

    #Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(), categorical_cols),
        ]
    )

    X_transformed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    #calculate MI
    mi_scores = mutual_info_classif(X_transformed, y, random_state=SEED)
    mi_df = pd.DataFrame({
        "Feature": feature_names,
        "MI_Score": mi_scores
    }).sort_values(by="MI_Score", ascending=False)

    #plots
    plt.figure(figsize=(18, 12))  # Increased figure size
    sns.set_context("poster", font_scale=0.9)
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("viridis", len(mi_df))
    ax = sns.barplot(
        data=mi_df,
        x="MI_Score",
        y="Feature",
        palette=palette,
        hue="Feature",
        legend=False
    )

    #Size en fontweight for the project
    plt.title("FEATURE IMPORTANCE: MUTUAL INFORMATION SCORES",
              fontsize=32, fontweight='bold', pad=30)

    plt.xlabel("Information Gain (Mutual Information Score)",
               fontsize=24, fontweight='bold', labelpad=20)

    plt.ylabel("Clinical Features (One-Hot Encoded)",
               fontsize=24, fontweight='bold', labelpad=20)

    # Bold tick labels
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')

    # Add data labels to the end of each bar (Bold and Bigger)
    for i, score in enumerate(mi_df["MI_Score"]):
        plt.text(score + 0.003, i, f"{score:.4f}",
                 va='center',
                 fontsize=18,
                 fontweight='bold',
                 color='black')

    plt.tight_layout()

    plot_path = RESULTS_DIR / "mutual_information_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    print(f"\n[SUCCESS] MI graph saved with bold text to: {plot_path}")

    # Save CSV
    mi_df.to_csv(RESULTS_DIR / "mi_scores_final.csv", index=False)


if __name__ == "__main__":
    main()