import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score,
    accuracy_score, classification_report
)

#Where to find th results
MODELS = {
    "KNN": "results/KNN/knn_test_predictions.csv",
    "SVM": "results/SVM/svm_test_predictions.csv",
    "Multivariate Logit": "results/Multivariate Logit/multivariate_logit_test_predictions.csv",
    "Random Forest": "results/RF/rf_test_predictions.csv",
    "XGBoost": "results/XGBoost/xgboost_test_predictions.csv"
}

#dic of the hyperpar
HYPERPARAMS = {
    "KNN": {
        "metric": "manhattan",
        "n_neighbors": 25,
        "p": 1,
        "weights": "distance"
    },
    "SVM": {
        "C": 10,
        "gamma": "auto",
        "kernel": "poly",
        "probability": True
    },
    "Multivariate Logit": {
        "C": 0.3593813663804626,
        "l1_ratio": 0.5,
        "max_iter": 2500,
        "penalty": "elasticnet",
        "solver": "saga"
    },
    "Random Forest": {
        "max_depth": 5,
        "max_features": "sqrt",
        "min_samples_split": 2,
        "n_estimators": 100
    },
    "XGBoost": {
        "eval_metric": "logloss",
        "learning_rate": 0.1,
        "max_depth": 2,
        "n_estimators": 100,
        "subsample": 0.8
    }
}

RESULTS_DIR = Path("results/Final_Comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    model_data = {}
    for name, path_str in MODELS.items():
        path = Path(path_str)
        if path.exists():
            model_data[name] = pd.read_csv(path)
        else:
            print(f"Warning: File {path} not found. Skipping {name}...")

    if not model_data:
        print("Error: No prediction files found. Ensure paths are correct relative to ML_Final.")
        return

    summary_list = []

    print("\n" + "=" * 95)
    print(f"{'MASTER EVALUATION: HEART DISEASE CLASSIFICATION':^95}")
    print("=" * 95)

    for name, df in model_data.items():
        y_true = df['y_true']
        y_pred = df['y_pred']
        y_proba = df['y_proba']

        # 1. Calculation of Metrics
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        summary_list.append({
            "Model": name,
            "AUC Score": roc_auc,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Recall (Sens)": recall_score(y_true, y_pred),
            "Specificity": specificity,
            "F1-Score": f1_score(y_true, y_pred)
        })

        # 2. Print Detailed Individual Report
        print(f"\n>>> MODEL: {name}")
        params = HYPERPARAMS.get(name, "Parameters not logged")
        print(f"Optimized Hyperparameters: {params}")
        print("-" * 95)
        print(classification_report(y_true, y_pred, target_names=['Healthy', 'Sick']))

        # Explicitly print the AUC and clinical metrics for the specific model
        print(f"DETAILED PERFORMANCE FOR {name}:")
        print(f"ROC-AUC SCORE:    {roc_auc:.4f}")
        print(f"SPECIFICITY:      {specificity:.4f}")
        print(f"CONFUSION MATRIX: [TP: {tp:3} | FP: {fp:3}]")
        print(f"                  [FN: {fn:3} | TN: {tn:3}]")
        print("-" * 95)

    # 3. Final Comparison Table
    summary_df = pd.DataFrame(summary_list).sort_values(by="AUC Score", ascending=False)

    print("\n" + "=" * 95)
    print(f"{'FINAL COMPARATIVE SUMMARY (Sorted by AUC)':^95}")
    print("=" * 95)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 95)



if __name__ == "__main__":
    main()