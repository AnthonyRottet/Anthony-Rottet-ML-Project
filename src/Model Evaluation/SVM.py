import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc

# --- SETTINGS ---
# Paths synchronized with project structure
PRED_PATH = Path("../../results/SVM/svm_test_predictions.csv")
RESULTS_DIR = Path("../../results/SVM")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not PRED_PATH.exists():
        print(f"Error: {PRED_PATH} not found. Please run the SVM script first!")
        return

    # 1. Load Data
    df = pd.read_csv(PRED_PATH)

    # Set visual style
    sns.set_theme(style="white")

    # 2. Confusion Matrix Heatmap
    # Using 'Blues' cmap for visual distinction of the SVM model
    cm = confusion_matrix(df['y_true'], df['y_pred'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy (0)', 'Sick (1)'],
                yticklabels=['Healthy (0)', 'Sick (1)'])

    plt.title('Confusion Matrix: SVM Model', fontsize=14)
    plt.ylabel('Actual Status', fontsize=12)
    plt.xlabel('Predicted Status', fontsize=12)

    plt.savefig(RESULTS_DIR / "svm_confusion_matrix.png", dpi=300)
    plt.close()

    # 3. ROC Curve Calculation
    # fpr = False Positive Rate, tpr = True Positive Rate
    fpr, tpr, thresholds = roc_curve(df['y_true'], df['y_proba'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='royalblue', lw=2, label=f'SVM ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve: SVM Performance', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.savefig(RESULTS_DIR / "svm_roc_curve.png", dpi=300)
    plt.close()

    # 4. Final Output Results
    print("\n=== SVM EVALUATION COMPLETE ===")
    print("-" * 45)
    print(f"SVM AUC    : {roc_auc:.4f}")
    print(f"Plots saved in : {RESULTS_DIR}")
    print("-" * 45)


if __name__ == "__main__":
    main()