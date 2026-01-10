import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

warnings.filterwarnings("ignore")

SEED = 42

DATA_PATH = Path("../../data/processed/heart_disease_clean_mini.csv")
RESULTS_DIR = Path("../../results/XGBoost")
PARAMS_OUT = RESULTS_DIR / "xgb_hyperparameters.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    #LOAD DATA
    df = pd.read_csv(DATA_PATH)
    target_col = "heart_disease"
    X = df.drop(target_col, axis=1)
    y = df[target_col].astype(int)

    #TRAIN-TEST SPLIT
    # Splitting to ensure tuning is validated on a hold-out set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    #DEFINE FEATURES
    numeric_cols = ["cholesterol", "max_hr", "st_depression"]
    categorical_cols = [
        "sex", "chest_pain_type", "ekg_results",
        "exercise_angina", "slope_of_st", "number_of_vessels_fluro", "thallium"
    ]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ])

    #GRID SEARCH SETUP
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [2, 3, 5],
        'model__subsample': [0.8, 1.0],
        'model__eval_metric': ['logloss', 'error', 'auc']
    }

    full_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", XGBClassifier(random_state=SEED))
    ])

    #Grid Search with CV=5
    clf = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )

    #EXECUTE SEARCH
    print("\n=== INITIATING XGBOOST GRID SEARCH (CV=5) ===")
    print("-" * 45)
    clf.fit(X_train, y_train)

    #SAVE BEST PARAMETERS TO CSV
    best_params = clf.best_params_
    params_df = pd.DataFrame([best_params])
    params_df.to_csv(PARAMS_OUT, index=False)
    print(f"[INFO] Best hyperparameters saved to: {PARAMS_OUT}")

    #EVALUATE BEST MODEL ON TEST SET
    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    #PRINT RESULTS
    print("\n--- TUNING COMPLETE ---")
    print(f"Best Parameters: {best_params}")
    print("-" * 45)
    print(f"1. Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"2. Test AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"3. Test F1-Score : {f1_score(y_test, y_pred):.4f}")
    print("-" * 45)


if __name__ == "__main__":
    main()