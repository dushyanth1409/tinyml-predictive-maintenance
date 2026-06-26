"""
Model Training
---------------
Trains and compares four classifiers on bearing fault features:
  - Random Forest
  - XGBoost
  - LightGBM
  - SVM (RBF kernel)

Uses stratified k-fold cross-validation and saves the best model.
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — skipping")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not installed — skipping")


FAULT_NAMES = ["Healthy", "Inner Race", "Outer Race", "Ball"]


def build_models() -> dict:
    """Return dict of model_name → sklearn Pipeline."""
    models = {
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=None,
                min_samples_split=2, random_state=42, n_jobs=-1
            ))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                        decision_function_shape="ovr", random_state=42))
        ]),
    }

    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=6,
                learning_rate=0.1, use_label_encoder=False,
                eval_metric="mlogloss", random_state=42,
                verbosity=0
            ))
        ])

    if HAS_LGB:
        models["LightGBM"] = Pipeline([
            ("clf", LGBMClassifier(
                n_estimators=200, max_depth=-1,
                learning_rate=0.1, num_leaves=31,
                random_state=42, verbose=-1
            ))
        ])

    return models


def train_and_evaluate(X: pd.DataFrame, y: np.ndarray,
                       n_folds: int = 5,
                       output_dir: str = "models/saved") -> pd.DataFrame:
    """
    Train all models with stratified k-fold CV and compare performance.

    Parameters
    ----------
    X          : pd.DataFrame  Feature matrix
    y          : np.ndarray    Integer labels
    n_folds    : int           Number of CV folds
    output_dir : str           Directory to save trained models

    Returns
    -------
    results : pd.DataFrame  Columns: model, accuracy, precision, recall, f1, train_time
    """
    os.makedirs(output_dir, exist_ok=True)
    models  = build_models()
    cv      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []

    print(f"\n{'Model':<16} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Time(s)':>10}")
    print("─" * 62)

    for name, model in models.items():
        t0 = time()
        cv_results = cross_validate(
            model, X, y, cv=cv,
            scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
            return_train_score=False
        )
        elapsed = time() - t0

        acc   = cv_results["test_accuracy"].mean()
        prec  = cv_results["test_precision_macro"].mean()
        rec   = cv_results["test_recall_macro"].mean()
        f1    = cv_results["test_f1_macro"].mean()

        print(f"{name:<16} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {elapsed:>10.2f}")

        # Refit on full dataset and save
        model.fit(X, y)
        joblib.dump(model, os.path.join(output_dir, f"{name}.pkl"))

        results.append({
            "model":     name,
            "accuracy":  round(acc, 4),
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "train_time_s": round(elapsed, 2),
        })

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    results_df.to_csv(os.path.join(output_dir, "comparison.csv"), index=False)

    best = results_df.iloc[0]["model"]
    print(f"\n✅  Best model: {best}  (F1 = {results_df.iloc[0]['f1']:.4f})")
    print(f"    Saved all models to: {output_dir}/")

    # Save metadata
    meta = {
        "best_model": best,
        "feature_names": list(X.columns),
        "classes": FAULT_NAMES,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return results_df
