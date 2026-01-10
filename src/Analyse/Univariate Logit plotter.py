import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path

# --- SETTINGS ---
# Path synchronized with your current ML_Final project structure
DATA_PATH = Path("../../data/processed/heart_disease_clean_mini.csv")
OUT_DIR = Path("../../results/Univariate Logit")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    target = "heart_disease"

    features = ["cholesterol", "max_hr", "st_depression", "age", "sex", "exercise_angina"]

    sns.set_theme(style="whitegrid")

    for feature in features:
        if feature not in df.columns:
            continue

        # Prepare data
        # Ensure y is integer for Logit math
        X_raw = df[[feature]].copy()
        y = df[target].astype(int)

        #Fit Model using statsmodels for the probability curve
        X_const = sm.add_constant(X_raw)
        try:
            model = sm.Logit(y, X_const).fit(disp=False)

            #Setup Plot
            plt.figure(figsize=(10, 6))
            sns.regplot(
                x=feature,
                y=target,
                data=df,
                logistic=True,
                ci=95,  # Shows confidence interval
                scatter_kws={'alpha': 0.4, 'color': 'steelblue'},
                line_kws={'color': 'crimson', 'lw': 3, 'label': 'Logit Curve'}
            )

            #Formatting
            plt.title(f"Univariate Risk Analysis: {feature.replace('_', ' ').title()}", fontsize=14)
            plt.ylabel("Probability of Heart Disease (0.0 - 1.0)", fontsize=12)
            plt.xlabel(f"Feature Value: {feature}", fontsize=12)
            plt.ylim(-0.05, 1.05)
            plt.grid(alpha=0.3)

            #Save
            file_name = f"logit_plot_{feature}.png"
            plt.savefig(OUT_DIR / file_name, dpi=300)
            plt.close()
            print(f"Successfully generated univariate plot for: {feature}")

        except Exception as e:
            print(f"Skipping {feature}: {e} (Likely perfect separation or non-numeric)")

    print(f"\n=== UNIVARIATE ANALYSIS COMPLETE ===")
    print(f"Plots saved in: {OUT_DIR.absolute()}")


if __name__ == "__main__":
    main()