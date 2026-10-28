"""
build_serving.py -- produit uniquement les artefacts necessaires a l'API,
sans comparaison de modeles, segmentation ni SHAP (utilise en CI).

Reprend les memes fonctions et seeds que la chaine complete :
1. regression logistique retenue (src.train) entrainee sur le train set
   -> best_model.pkl, test_split.pkl ;
2. recalibration CalibratedClassifierCV(cv=5, ensemble=False), methode lue
   dans reports/robustness/summary.json (sigmoid par defaut) -> calibrated_model.pkl ;
3. seuil de la matrice de couts sur les probabilites hors-fold du train
   (src.business, hypotheses par defaut) -> serving_config.json.

Usage : python -m src.build_serving [--model-dir models]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from src.business import Costs, cost_threshold_from_oof, write_serving_config
from src.preprocessing import build_preprocessor, clean_data, load_data, split_X_y
from src.train import (
    RANDOM_STATE,
    REFERENCE_MODEL,
    TEST_SIZE,
    build_pipeline,
    compute_scale_pos_weight,
    make_models,
)

ROBUSTNESS_SUMMARY = Path("reports/robustness/summary.json")


def calibration_method(summary_path: Path = ROBUSTNESS_SUMMARY) -> str:
    """Methode retenue par src.robustness ; sigmoid si absente."""
    if summary_path.exists():
        method = json.loads(summary_path.read_text())["calibration"].get("chosen_method")
        if method in ("sigmoid", "isotonic"):
            return method
    return "sigmoid"


def build(data: str, model_dir: Path, method: str, costs: Costs | None = None) -> dict:
    costs = costs or Costs()
    model_dir.mkdir(parents=True, exist_ok=True)
    df = clean_data(load_data(data))
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    model = make_models(compute_scale_pos_weight(y_train))[REFERENCE_MODEL]
    served = build_pipeline(model, build_preprocessor(df)).fit(X_train, y_train)
    calibrated = CalibratedClassifierCV(clone(served), method=method, cv=5, ensemble=False, n_jobs=1)
    calibrated.fit(X_train, y_train)
    threshold, _ = cost_threshold_from_oof(calibrated, X_train, y_train, costs)

    joblib.dump(served, model_dir / "best_model.pkl")
    joblib.dump(calibrated, model_dir / "calibrated_model.pkl")
    joblib.dump({"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test},
                model_dir / "test_split.pkl")
    config_path = write_serving_config(model_dir, model_dir / "calibrated_model.pkl", threshold, costs)
    return {"threshold": threshold, "method": method, "config": config_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/telco_churn.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    t0 = time.time()
    out = build(args.data, Path(args.model_dir), calibration_method())
    print(f"Artefacts de service ecrits dans {args.model_dir}/ (calibration {out['method']}, "
          f"seuil {out['threshold']:.2f}) en {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
