"""
evaluate.py -- evalue le modele sauvegarde par train.py sur le test set
mis de cote (jamais vu pendant la validation croisee), et produit les
metriques + figures (matrice de confusion, courbe ROC).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--fig-dir", default="reports/figures")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pipeline = joblib.load(model_dir / "best_model.pkl")
    split = joblib.load(model_dir / "test_split.pkl")
    X_test, y_test = split["X_test"], split["y_test"]

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])

    print("Metriques sur le test set (holdout, jamais vu en CV) :")
    for k, v in metrics.items():
        print(f"  {k:10s} {v:.4f}")
    print("\n" + report)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No Churn", "Churn"], cmap="Blues", ax=ax
    )
    ax.set_title("Matrice de confusion -- test set")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("Courbe ROC -- test set")
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve.png", dpi=120)
    plt.close(fig)

    with open(model_dir / "test_metrics.json", "w") as f:
        json.dump({"metrics": metrics, "classification_report": report}, f, indent=2)

    print(f"\nFigures sauvegardees dans {fig_dir}/")


if __name__ == "__main__":
    main()
