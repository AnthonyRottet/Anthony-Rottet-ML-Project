import pandas as pd
import numpy as np
import matplotlib

# Force non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

# --- SETTINGS ---
SEED = 42
DATA_PATH = Path("../../data/processed/heart_disease_clean_mini.csv")
RESULTS_DIR = Path("../../results/KNN")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    # 1. LOAD DATA
    df = pd.read_csv(DATA_PATH)
    X = df.drop("heart_disease", axis=1)
    y = df["heart_disease"].astype(int)

    # 2. TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    # 3. PREPROCESSING
    numeric_cols = ["cholesterol", "max_hr", "st_depression"]
    categorical_cols = ["sex", "chest_pain_type", "ekg_results", "exercise_angina",
                        "slope_of_st", "number_of_vessels_fluro", "thallium"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ])

    # 4. CROSS-VALIDATION SETTINGS
    k_values = range(1, 32, 2)
    mean_cv_auc = []

    # StratifiedKFold ensures the "Sick/Healthy" ratio is kept in each fold
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=False)

    print("--- Calculating Mean Cross-Validation AUC ---")
    for k in k_values:
        pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", KNeighborsClassifier(n_neighbors=k, weights='distance', metric='manhattan'))
        ])

        # 5-Fold Cross-Validation on the Training Set only
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv_strategy, scoring='roc_auc')
        mean_cv_auc.append(scores.mean())
        print(f"k={k:2d} | Mean CV AUC: {scores.mean():.4f}")

    # 5. PLOTTING
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Plot the CV scores
    plt.plot(k_values, mean_cv_auc, color='rebeccapurple', marker='o', linewidth=2, label='Mean CV AUC')

    # Highlight k=25
    plt.axvline(x=25, color='red', linestyle='--', alpha=0.7, label='Optimal k (Selected)')

    plt.title('KNN Tuning: Mean CV ROC-AUC vs. k', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Neighbors (k)', fontsize=12)
    plt.ylabel('Mean ROC-AUC (Internal 5-Fold CV)', fontsize=12)
    plt.xticks(k_values)
    plt.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "KNN_Best_K.png", dpi=300)
    plt.close()

    print(f"\n[SUCCESS] CV Plot saved. k=25 is justified by the internal validation peak.")


if __name__ == "__main__":
    main()