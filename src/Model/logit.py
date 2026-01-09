import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

DATA_PATH = Path("../../data/processed/heart_disease_clean_mini.csv")
OUT_PATH = Path("../../results/LOGIT/univariate_logit_results.csv")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def run_univariate_logit(df, target_col):
    results = []
    features = [c for c in df.columns if c != target_col]

    for feature in features:
        X = df[[feature]]
        y = df[target_col]

        # Skip constant features
        if X[feature].nunique() <= 1:
            results.append({
                "feature": feature,
                "coef": None,
                "odds_ratio": None,
                "ci95_low": None,
                "ci95_high": None,
                "p_value": None,
                "pseudo_r2": None,
                "error": "constant feature"
            })
            continue

        X = sm.add_constant(X, has_constant="add")

        try:
            model = sm.Logit(y, X).fit(disp=False)

            coef = float(model.params[feature])
            se = float(model.bse[feature])
            pval = float(model.pvalues[feature])
            pseudo_r2 = float(model.prsquared)

            # 95% CI on coef, then exponentiate to get OR CI
            ci_low = coef - 1.96 * se
            ci_high = coef + 1.96 * se

            results.append({
                "feature": feature,
                "coef": coef,
                "odds_ratio": float(np.exp(coef)),
                "ci95_low": float(np.exp(ci_low)),
                "ci95_high": float(np.exp(ci_high)),
                "p_value": pval,
                "pseudo_r2": pseudo_r2,
                "error": None
            })

        except Exception as e:
            results.append({
                "feature": feature,
                "coef": None,
                "odds_ratio": None,
                "ci95_low": None,
                "ci95_high": None,
                "p_value": None,
                "pseudo_r2": None,
                "error": str(e)
            })

    return pd.DataFrame(results)


def main():
    print("Running univariate logistic regressions...")

    df = pd.read_csv(DATA_PATH)
    target_col = "heart_disease"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    results = run_univariate_logit(df, target_col)

    # Sort by p-value (most significant first)
    results = results.sort_values("p_value", na_position="last")

    # Save
    results.to_csv(OUT_PATH, index=False)

    # Print top 10 nicely
    print("\n=== TOP RESULTS (by p-value) ===")
    cols = ["feature", "coef", "odds_ratio", "ci95_low", "ci95_high", "p_value", "pseudo_r2", "error"]
    print(results[cols].head(10).to_string(index=False))

    print("\nSaved results to:", OUT_PATH)


if __name__ == "__main__":
    main()
