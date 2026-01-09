import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc

# --- SETTINGS ---
# Paths synchronized with project structure
PRED_PATH = Path("../../results/RF/rf_test_predictions.csv")
RESULTS_DIR = Path("../../results/RF")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not PRED_PATH.exists():
        print(f"Error: {PRED_PATH} not found. Please run the Random Forest script first!")
        return

    # 1. Load Data
    df = pd.read_csv(PRED_PATH)

    # Set visual style
    sns.set_theme(style="white")

    # 2. Confusion Matrix Heatmap
    # Using 'Greens' cmap to distinguish RF from other models
    cm = confusion_matrix(df['y_true'], df['y_pred'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Healthy (0)', 'Sick (1)'],
                yticklabels=['Healthy (0)', 'Sick (1)'])

    plt.title('Confusion Matrix: Random Forest Model', fontsize=14)
    plt.ylabel('Actual Status', fontsize=12)
    plt.xlabel('Predicted Status', fontsize=12)

    plt.savefig(RESULTS_DIR / "rf_confusion_matrix.png", dpi=300)
    plt.close()

    # 3. ROC Curve Calculation
    # fpr = False Positive Rate, tpr = True Positive Rate
    fpr, tpr, thresholds = roc_curve(df['y_true'], df['y_proba'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='forestgreen', lw=2, label=f'RF ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve: Random Forest Performance', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.savefig(RESULTS_DIR / "rf_roc_curve.png", dpi=300)
    plt.close()

    # 4. Final Output Results
    print("\n=== RANDOM FOREST EVALUATION COMPLETE ===")
    print("-" * 45)
    print(f"Random Forest AUC : {roc_auc:.4f}")
    print(f"Plots saved in     : {RESULTS_DIR}")
    print("-" * 45)


if __name__ == "__main__":
    main()