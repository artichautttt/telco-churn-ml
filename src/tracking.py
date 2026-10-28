"""
tracking.py -- suivi MLflow des entrainements (tracking local, file:./mlruns,
sans serveur dedie).

Structure d'une execution de train.py dans l'experience "telco-churn" :
- un run parent "model_comparison" : parametres globaux (split, CV, seed),
  nom du modele retenu, et pipeline final (ColumnTransformer + modele)
  logue comme modele MLflow ;
- un run enfant par modele compare : hyperparametres + metriques de CV
  (moyenne et ecart-type sur les 5 folds) + metriques sur le test set.

Le tracking ne doit jamais faire echouer l'entrainement : si l'ecriture
echoue, un avertissement est affiche et train.py se termine normalement
(best_model.pkl est deja sauvegarde a ce stade).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "telco-churn"


def log_training_run(
    *,
    cv_results: dict,
    models: dict,
    best_name: str,
    best_pipeline,
    input_example,
    run_params: dict,
    test_results: dict | None = None,
    justification: str | None = None,
) -> None:
    """Enregistre une execution de train.py dans MLflow ; ne leve jamais d'exception."""
    try:
        _log_training_run(
            cv_results=cv_results,
            models=models,
            best_name=best_name,
            best_pipeline=best_pipeline,
            input_example=input_example,
            run_params=run_params,
            test_results=test_results or {},
            justification=justification,
        )
    except Exception:
        logger.warning("echec du tracking MLflow ; l'entrainement reste valide.", exc_info=True)


def _log_training_run(
    *, cv_results, models, best_name, best_pipeline, input_example, run_params,
    test_results, justification,
):
    # mlflow >= 3 exige cet opt-in explicite pour le backend fichier local.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    import mlflow
    import mlflow.sklearn

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="model_comparison") as parent:
        mlflow.log_params(run_params)

        for name, metrics in cv_results.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tag("model_name", name)
                mlflow.log_params(models[name].get_params())
                mlflow.log_metrics({
                    f"cv_{metric}_{stat}": value
                    for metric, stats in metrics.items()
                    for stat, value in stats.items()
                })
                if name in test_results:
                    mlflow.log_metrics({f"test_{k}": v for k, v in test_results[name].items()})

        mlflow.set_tag("best_model", best_name)
        if justification:
            mlflow.set_tag("selection_justification", justification)
        if best_name in test_results:
            mlflow.log_metrics({f"best_test_{k}": v for k, v in test_results[best_name].items()})
        mlflow.log_metrics({
            f"best_cv_{metric}_mean": stats["mean"]
            for metric, stats in cv_results[best_name].items()
        })
        mlflow.sklearn.log_model(best_pipeline, name="model", input_example=input_example)

    print(f"Run MLflow '{parent.info.run_id}' logue dans {tracking_uri} "
          f"(experience '{EXPERIMENT_NAME}')")
