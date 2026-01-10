import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc

MODELS = {
    "KNN": "../../results/KNN/knn_test_predictions.csv",
    "SVM": "../../results/SVM/svm_test_predictions.csv",
    "Multivariate Logit": "../../results/Multivariate Logit/multivariate_logit_test_predictions.csv",
    "Random Forest": "../../results/RF/rf_test_predictions.csv",
    "XGBoost": "../../results/XGBoost/xgboost_test_predictions.csv"
}

COLORS = {
    "KNN": "blue",
    "SVM": "royalblue",
    "Multivariate Logit": "darkorange",
    "Random Forest": "forestgreen",
    "XGBoost": "red"
}

RESULTS_DIR = Path("../../results/Final_Comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    #Load the data and check files
    model_data = {}
    for name, path_str in MODELS.items():
        path = Path(path_str)
        if path.exists():
            model_data[name] = pd.read_csv(path)
        else:
            print(f"Warning: {path} not found. Skipping {name}...")

    if not model_data:
        print("Error: No prediction files found. Run the classifier scripts first!")
        return

    # et visual style
    sns.set_theme(style="white")

    #Plot Combined ROC Curves
    plt.figure(figsize=(10, 8))

    for name, df in model_data.items():
        fpr, tpr, _ = roc_curve(df['y_true'], df['y_proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=COLORS[name], lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Combined Model Comparison: ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.savefig(RESULTS_DIR / "combined_roc_comparison.png", dpi=300)
    plt.close()

    #Plot Confusion Matrix Grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (name, df) in enumerate(model_data.items()):
        cm = confusion_matrix(df['y_true'], df['y_pred'])

        sns.heatmap(cm,
                    annot=True,
                    fmt='d',
                    cmap='Purples',
                    ax=axes[i],
                    annot_kws={"size": 30, "weight": "bold"},  # Large bold numbers inside boxes
                    xticklabels=['Healthy', 'Sick'],
                    yticklabels=['Healthy', 'Sick'],
                    cbar=False)  # Removed colorbar to give more space to the matrix

        # Style the tick labels (Healthy/Sick)
        axes[i].set_xticklabels(['Healthy', 'Sick'], fontsize=14, fontweight='bold')
        axes[i].set_yticklabels(['Healthy', 'Sick'], fontsize=14, fontweight='bold')
        axes[i].set_title(f'{name}', fontsize=25, fontweight='bold', pad=15)
        axes[i].set_ylabel('Actual Status', fontsize=20, fontweight='bold')
        axes[i].set_xlabel('Predicted Status', fontsize=20, fontweight='bold')

    # Remove the 6th plot
    if len(model_data) < 6:
        fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix_grid.png", dpi=300)
    plt.close()

    #Performance Summary Table
    summary_list = []
    for name, df in model_data.items():
        fpr, tpr, _ = roc_curve(df['y_true'], df['y_proba'])
        summary_list.append({
            "Model": name,
            "AUC Score": auc(fpr, tpr),
            "Accuracy": (df['y_true'] == df['y_pred']).mean()
        })

    summary_df = pd.DataFrame(summary_list).sort_values(by="AUC Score", ascending=False)
    summary_df.to_csv(RESULTS_DIR / "model_performance_summary.csv", index=False)

    print("\n=== MASTER EVALUATION COMPLETE ===")
    print("-" * 45)
    print("Performance Summary (Sorted by AUC):")
    print(summary_df.to_string(index=False))
    print("-" * 45)
    print(f"Comparison plots saved in: {RESULTS_DIR.absolute()}")
    print("-" * 45)


if __name__ == "__main__":
    main()