"""
log_analysis.py -- logue les resultats des etapes 3 a 6 dans MLflow
(experience "telco-churn", run "advanced_analysis") : metriques extraites des
fichiers reports/*/summary.json, et rapports/figures en artefacts.

Comme src.tracking, ne fait jamais echouer la chaine : un echec MLflow
affiche un avertissement.

Usage : python -m src.log_analysis
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from src.tracking import DEFAULT_TRACKING_URI, EXPERIMENT_NAME

logger = logging.getLogger(__name__)
REPORTS = Path("reports")


def collect_metrics(reports: Path = REPORTS) -> dict[str, float]:
    """Aplati les metriques numeriques cles des summary.json en {nom: valeur}."""
    m: dict[str, float] = {}

    def load(name):
        p = reports / name / "summary.json"
        return json.loads(p.read_text()) if p.exists() else None

    if seg := load("segmentation"):
        m |= {"seg_k_kprototypes": seg["k_kprototypes"], "seg_k_kmeans": seg["k_kmeans"],
              "seg_adjusted_rand_index": seg["adjusted_rand_index"]}
    if inter := load("interactions"):
        auc = inter["logistic_cv_auc"]
        m |= {"lr_cv_auc_baseline": auc["baseline_mean"],
              "lr_cv_auc_with_interactions": auc["with_interaction_terms_mean"]}
        for i, row in enumerate(inter["top_interactions"][:3], 1):
            m[f"interaction_top{i}_strength"] = row["strength"]
    if rob := load("robustness"):
        for model, r in rob["bootstrap"].items():
            for metric in ("auc", "recall"):
                m[f"boot_{model}_{metric}_ci_low"] = r[metric]["ci95"][0]
                m[f"boot_{model}_{metric}_ci_high"] = r[metric]["ci95"][1]
        for k, r in rob["shap_stability"].items():
            m[f"shap_top10_spearman_mean_{k}"] = r["spearman_mean"]
            m[f"shap_top10_overlap_mean_{k}"] = r["top_overlap_mean"]
        for k, v in rob["calibration"]["brier_test"].items():
            m[f"brier_test_{k}"] = v
    if bus := load("business"):
        m |= {"cost_optimal_threshold": bus["optimal_threshold_train_oof"],
              "cost_test_at_optimal": bus["at_optimal_threshold"]["total_cost_test"],
              "cost_test_at_0_5": bus["at_default_threshold_0_5"]["total_cost_test"],
              "recall_test_at_optimal": bus["at_optimal_threshold"]["recall"],
              "capture_top10": bus["capture_top10"], "capture_top20": bus["capture_top20"],
              "lift_top10": bus["lift_top10"], "lift_top20": bus["lift_top20"]}
    return {k: float(v) for k, v in m.items()}


def main():
    metrics = collect_metrics()
    if not metrics:
        print("Aucun summary.json trouve dans reports/ : rien a loguer.")
        return
    try:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name="advanced_analysis") as run:
            mlflow.log_metrics(metrics)
            for d in ("segmentation", "interactions", "robustness", "business"):
                if (REPORTS / d).exists():
                    mlflow.log_artifacts(str(REPORTS / d), artifact_path=d)
            if (REPORTS / "figures").exists():
                mlflow.log_artifacts(str(REPORTS / "figures"), artifact_path="figures")
        print(f"{len(metrics)} metriques loguees dans le run MLflow {run.info.run_id}")
    except Exception:
        logger.warning("echec du logging MLflow de l'analyse", exc_info=True)


if __name__ == "__main__":
    main()
