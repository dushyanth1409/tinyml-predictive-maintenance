"""
Model Evaluation
-----------------
Detailed evaluation: confusion matrix, per-class metrics, feature importance.
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

FAULT_NAMES = ["Healthy", "Inner Race", "Outer Race", "Ball"]


def evaluate_model(model_path: str, X: pd.DataFrame, y: np.ndarray,
                   test_size: float = 0.2, plot: bool = True) -> dict:
    """
    Load a saved model and run full evaluation on a held-out test set.

    Returns
    -------
    dict with accuracy, per-class metrics, confusion matrix
    """
    model = joblib.load(model_path)
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred,
                                   target_names=FAULT_NAMES,
                                   output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'─'*50}")
    print(classification_report(y_test, y_pred, target_names=FAULT_NAMES))

    if plot:
        _plot_evaluation(model, X_test, y_test, y_pred, cm, model_name)

    return {"report": report, "confusion_matrix": cm.tolist(), "model": model_name}


def _plot_evaluation(model, X_test, y_test, y_pred, cm, model_name):
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f"Bearing Fault Detection — {model_name}", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    # Confusion matrix
    ax1 = fig.add_subplot(gs[0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=FAULT_NAMES)
    disp.plot(ax=ax1, colorbar=False, cmap="Blues")
    ax1.set_title("Confusion Matrix")
    ax1.tick_params(axis='x', rotation=30)

    # Feature importance (if available)
    ax2 = fig.add_subplot(gs[1])
    clf = model.named_steps.get("clf", model)
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        feature_names = list(X_test.columns)
        top_idx = np.argsort(importances)[::-1][:10]
        ax2.barh([feature_names[i] for i in top_idx[::-1]],
                 importances[top_idx[::-1]], color="#2196F3")
        ax2.set_title("Top 10 Feature Importances")
        ax2.set_xlabel("Importance")
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "Feature importance\nnot available for this model",
                 ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    plt.show()


def compare_all(models_dir: str, X: pd.DataFrame, y: np.ndarray):
    """Load and compare all saved models side by side."""
    pkl_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    if not pkl_files:
        print(f"No .pkl files found in {models_dir}")
        return

    comparison_csv = os.path.join(models_dir, "comparison.csv")
    if os.path.exists(comparison_csv):
        df = pd.read_csv(comparison_csv)
        print("\n── Saved CV Results ──────────────────────────────────────")
        print(df.to_string(index=False))
