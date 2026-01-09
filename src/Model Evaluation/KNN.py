import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path

# Path to the KNN predictions (matching your results folder)
PRED_PATH = Path("../../results/KNN/knn_test_predictions.csv")
OUT_DIR = Path("../../results/KNN")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not PRED_PATH.exists():
        print(f"Error: Could not find {PRED_PATH}. Please run the knn_model.py script first!")
        return

    #Load KNN test predictions
    df = pd.read_csv(PRED_PATH)

    # Set visual style for a professional look
    sns.set_theme(style="white")


    #Confusion Matrix Heatmap

    cm = confusion_matrix(df['y_true'], df['y_pred'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Healthy (0)', 'Sick (1)'],
                yticklabels=['Healthy (0)', 'Sick (1)'])
    plt.title('Confusion Matrix: K-Nearest Neighbors')
    plt.ylabel('Actual Status')
    plt.xlabel('Predicted Status')
    plt.savefig(OUT_DIR / "knn_confusion_matrix.png")
    plt.close()


    #ROC Curve

    fpr, tpr, thresholds = roc_curve(df['y_true'], df['y_proba'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='rebeccapurple', lw=2, label=f'KNN ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve: KNN Performance')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(OUT_DIR / "knn_roc_curve.png")
    plt.close()

    print(f"--- KNN EVALUATION COMPLETE ---")
    print(f"Plots saved in: {OUT_DIR.absolute()}")
    print(f"KNN AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()