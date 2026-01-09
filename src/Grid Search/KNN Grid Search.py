import pandas as pd
import numpy as np
import warnings
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Suppress minor warnings
warnings.filterwarnings("ignore")

# --- SETTINGS ---
SEED = 42
DATA_PATH = Path("../../data/processed/heart_disease_clean_mini.csv")
RESULTS_DIR = Path("../../results/KNN")
PARAMS_OUT = RESULTS_DIR / "knn_hyperparameters.csv"

# Ensure the results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    # 1. LOAD DATA
    df = pd.read_csv(DATA_PATH)
    target_col = "heart_disease"
    X = df.drop(target_col, axis=1)
    y = df[target_col].astype(int)

    # 2. TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    # 3. DEFINE FEATURES
    numeric_cols = ["cholesterol", "max_hr", "st_depression"]
    categorical_cols = [
        "sex", "chest_pain_type", "ekg_results",
        "exercise_angina", "slope_of_st", "number_of_vessels_fluro", "thallium"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ]
    )

    # 4. GRID SEARCH SETUP
    full_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", KNeighborsClassifier())
    ])

    param_grid = {
        'model__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['euclidean', 'manhattan'],
        'model__p': [1, 2]
    }

    # Grid Search with CV=5
    clf = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )

    # 5. EXECUTE SEARCH
    print("\n INITIATING KNN GRID SEARCH (CV=5) ")
    print("-" * 45)
    clf.fit(X_train, y_train)

    # 6. SAVE BEST PARAMETERS TO CSV
    # Convert dictionary to DataFrame for CSV export
    best_params = clf.best_params_
    params_df = pd.DataFrame([best_params])
    params_df.to_csv(PARAMS_OUT, index=False)
    print(f"[INFO] Best hyperparameters saved to: {PARAMS_OUT}")

    # 7. EVALUATE BEST MODEL ON TEST SET
    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # 8. PRINT RESULTS
    print("\n TUNING COMPLETE ")
    print(f"Best Parameters: {best_params}")
    print("-" * 45)
    print(f"1. Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"2. Test AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"3. Test F1-Score : {f1_score(y_test, y_pred):.4f}")
    print("-" * 45)


if __name__ == "__main__":
    main()