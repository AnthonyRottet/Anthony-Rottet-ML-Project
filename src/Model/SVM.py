import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report
)

SEED = 42

# Paths
DATA_PATH = Path("../../data/processed/heart_disease_clean_mini.csv")
RESULTS_DIR = Path("../../results/SVM")
PRED_OUT = RESULTS_DIR / "svm_test_predictions.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    target_col = "heart_disease"

    #Define Features
    numeric_cols = ["cholesterol", "max_hr", "st_depression"]
    categorical_cols = [
        "sex", "chest_pain_type", "ekg_results",
        "exercise_angina", "slope_of_st", "number_of_vessels_fluro", "thallium"
    ]

    X = df[numeric_cols + categorical_cols]
    y = df[target_col].astype(int)

    #Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=SEED,
        stratify=y
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop='first'), categorical_cols),
        ]
    )
    svm_model = SVC(
        kernel='poly',
        C=10,
        gamma='auto',
        probability=True,
        class_weight='balanced',
        random_state=SEED
    )

    #Pipeline
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", svm_model)
    ])

    #Fit & Evaluate
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = clf.predict(X_test)

    print("\n SUPPORT VECTOR MACHINE (SVM) FINAL RESULTS")
    print("-" * 45)
    print(f"Test AUC : {roc_auc_score(y_test, proba):.4f}")
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print("-" * 45)
    print("\nClassification Report:\n", classification_report(y_test, pred))

    #Save Results
    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_proba"] = proba
    pred_df["y_pred"] = pred
    pred_df.to_csv(PRED_OUT, index=False)

    print(f"\nSaved SVM predictions to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()