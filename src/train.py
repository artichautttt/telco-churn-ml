"""
train.py -- entraine et compare trois modeles (regression logistique,
random forest, XGBoost) sur le dataset Telco Customer Churn, a reequilibrage
des classes egal, en validation croisee stratifiee 5-fold puis sur le test
set ; applique la regle de selection (voir `select_model`) et sauvegarde le
pipeline retenu (servi par l'API) ainsi que le pipeline XGBoost (utilise
pour SHAP TreeExplainer).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor, clean_data, load_data, split_X_y
from src.tracking import log_training_run

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
REFERENCE_MODEL = "logistic_regression"
# Regle de selection : la regression logistique (simple, interpretable, deja
# servie) est conservee si son ecart d'AUC en CV avec le meilleur modele est
# < AUC_TOLERANCE et si son rappel churn en CV est au moins equivalent
# (>= rappel du meilleur modele - RECALL_TOLERANCE).
AUC_TOLERANCE = 0.01
RECALL_TOLERANCE = 0.01

# Metriques sur la classe churn (pos_label=1, seuil 0,5) + accuracy.
SCORING = {
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "accuracy": "accuracy",
}
# Precision / rappel par classe (0 = No Churn, 1 = Churn), traces dans MLflow.
PER_CLASS_SCORING = {
    f"{name}_class{label}": make_scorer(metric, pos_label=label, zero_division=0)
    for name, metric in (("precision", precision_score), ("recall", recall_score))
    for label in (0, 1)
}


def compute_scale_pos_weight(y) -> float:
    """Ratio negatifs / positifs, equivalent XGBoost de class_weight='balanced'."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        raise ValueError("aucun exemple positif : scale_pos_weight indefini")
    return float((y == 0).sum() / n_pos)


def make_models(scale_pos_weight: float) -> dict:
    """Les trois modeles, a reequilibrage egal des classes."""
    return {
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
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_pipeline(model, preprocessor) -> Pipeline:
    return Pipeline([("preprocessor", clone(preprocessor)), ("model", clone(model))])


def cross_validate_models(X_train, y_train, preprocessor, models: dict) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = SCORING | PER_CLASS_SCORING
    results = {}
    for name, model in models.items():
        scores = cross_validate(
            build_pipeline(model, preprocessor), X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        results[name] = {
            metric: {
                "mean": float(np.mean(scores[f"test_{metric}"])),
                "std": float(np.std(scores[f"test_{metric}"])),
            }
            for metric in scoring
        }
    return results


def evaluate_on_test(pipeline, X_test, y_test) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = pipeline.predict(X_test)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "recall": float(recall_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred)),
        "accuracy": float(accuracy_score(y_test, pred)),
    }


def select_model(
    cv_results: dict,
    reference: str = REFERENCE_MODEL,
    auc_tol: float = AUC_TOLERANCE,
    recall_tol: float = RECALL_TOLERANCE,
) -> tuple[str, str]:
    """Applique la regle de selection sur les moyennes de CV (le test set
    reste reserve a l'evaluation). Retourne (modele_retenu, justification)."""
    auc = {n: r["roc_auc"]["mean"] for n, r in cv_results.items()}
    recall = {n: r["recall"]["mean"] for n, r in cv_results.items()}
    best_auc = max(auc, key=auc.get)
    gap = auc[best_auc] - auc[reference]
    recall_ok = recall[reference] >= recall[best_auc] - recall_tol

    if gap < auc_tol and recall_ok:
        return reference, (
            f"{reference} conserve : ecart d'AUC CV avec le meilleur ({best_auc}) = "
            f"{gap:.4f} < {auc_tol} et rappel churn CV {recall[reference]:.4f} >= "
            f"{recall[best_auc]:.4f} - {recall_tol}"
        )
    return best_auc, (
        f"{best_auc} retenu : ecart d'AUC CV avec {reference} = {gap:.4f} "
        f"(seuil {auc_tol}), rappel churn CV {recall[best_auc]:.4f} vs "
        f"{recall[reference]:.4f} (tolerance {recall_tol})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/telco_churn.csv")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    df = clean_data(load_data(args.data))
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    spw = compute_scale_pos_weight(y_train)
    models = make_models(spw)
    preprocessor = build_preprocessor(df)
    print(f"scale_pos_weight XGBoost (calcule sur le train set) = {spw:.4f}")

    print("Validation croisee (5-fold stratifie) sur le train set...")
    cv_results = cross_validate_models(X_train, y_train, preprocessor, models)

    fitted, test_results = {}, {}
    for name, model in models.items():
        fitted[name] = build_pipeline(model, preprocessor).fit(X_train, y_train)
        test_results[name] = evaluate_on_test(fitted[name], X_test, y_test)

    print(f"\n{'modele':20s} {'AUC cv':>8s} {'rap cv':>8s} {'prec cv':>8s} {'F1 cv':>8s} "
          f"{'AUC test':>9s} {'rap test':>9s} {'prec test':>9s} {'F1 test':>8s}")
    for name in models:
        c, t = cv_results[name], test_results[name]
        print(f"{name:20s} {c['roc_auc']['mean']:8.4f} {c['recall']['mean']:8.4f} "
              f"{c['precision']['mean']:8.4f} {c['f1']['mean']:8.4f} {t['roc_auc']:9.4f} "
              f"{t['recall']:9.4f} {t['precision']:9.4f} {t['f1']:8.4f}")

    best_name, justification = select_model(cv_results)
    print(f"\nDecision : {justification}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    joblib.dump(fitted[best_name], out_dir / "best_model.pkl")
    joblib.dump(fitted["xgboost"], out_dir / "xgboost_model.pkl")
    joblib.dump(
        {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test},
        out_dir / "test_split.pkl",
    )
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump(
            {
                "cv_results": cv_results,
                "test_results": test_results,
                "best_model": best_name,
                "justification": justification,
                "scale_pos_weight": spw,
            },
            f,
            indent=2,
        )
    print(f"\nModele retenu sauvegarde dans {out_dir / 'best_model.pkl'}")

    log_training_run(
        cv_results=cv_results,
        test_results=test_results,
        models=models,
        best_name=best_name,
        best_pipeline=fitted[best_name],
        input_example=X_train.head(3),
        run_params={
            "data": args.data,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "xgb_scale_pos_weight": round(spw, 6),
            "selection_rule": (
                f"keep {REFERENCE_MODEL} if cv_auc_gap<{AUC_TOLERANCE} "
                f"and cv_recall>=best-{RECALL_TOLERANCE}"
            ),
        },
        justification=justification,
    )


if __name__ == "__main__":
    main()
