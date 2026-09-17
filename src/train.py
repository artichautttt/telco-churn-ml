"""
train.py -- entraine et compare plusieurs modeles scikit-learn (regression
logistique, random forest, XGBoost) sur le dataset Telco Customer Churn,
avec validation croisee, puis sauvegarde le meilleur pipeline entraine.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor, clean_data, load_data, split_X_y

RANDOM_STATE = 42
SCORING = ["accuracy", "f1", "roc_auc"]

MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


def build_pipeline(model_name: str, preprocessor) -> Pipeline:
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", MODELS[model_name]),
    ])


def cross_validate_models(X_train, y_train, preprocessor) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name in MODELS:
        pipe = build_pipeline(name, preprocessor)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        results[name] = {
            metric: {
                "mean": float(np.mean(scores[f"test_{metric}"])),
                "std": float(np.std(scores[f"test_{metric}"])),
            }
            for metric in SCORING
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/telco_churn.csv")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    df = clean_data(load_data(args.data))
    X, y = split_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(df)

    print("Validation croisee (5-fold stratifie) sur le train set...")
    cv_results = cross_validate_models(X_train, y_train, preprocessor)
    for name, metrics in cv_results.items():
        print(f"\n{name}")
        for metric, stats in metrics.items():
            print(f"  {metric:10s} mean={stats['mean']:.4f}  std={stats['std']:.4f}")

    best_name = max(cv_results, key=lambda n: cv_results[n]["roc_auc"]["mean"])
    print(f"\nMeilleur modele (ROC-AUC en CV) : {best_name}")

    best_pipeline = build_pipeline(best_name, preprocessor)
    best_pipeline.fit(X_train, y_train)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    joblib.dump(best_pipeline, out_dir / "best_model.pkl")
    joblib.dump({"X_test": X_test, "y_test": y_test}, out_dir / "test_split.pkl")

    with open(out_dir / "cv_results.json", "w") as f:
        json.dump({"cv_results": cv_results, "best_model": best_name}, f, indent=2)

    print(f"\nModele sauvegarde dans {out_dir / 'best_model.pkl'}")


if __name__ == "__main__":
    main()
