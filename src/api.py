"""
api.py -- API FastAPI de serving du modele de churn.

Le modele et le seuil de decision sont lus dans models/serving_config.json,
ecrit par `python -m src.business` :
- modele : regression logistique recalibree (Platt, CalibratedClassifierCV
  ensemble=False) autour du pipeline complet (ColumnTransformer + modele) ;
  le pretraitement applique en serving est donc celui de l'entrainement, et
  la probabilite renvoyee est calibree ;
- seuil : optimise par la matrice de couts (etape 6) sur les probabilites
  calibrees hors-fold du train -- pas 0,5.

Lancement : uvicorn src.api:app --reload
Variables d'environnement : SERVING_CONFIG (defaut models/serving_config.json),
MODEL_PATH (remplace le modele de la configuration).
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from src.schemas import CustomerFeatures, PredictionResponse

DEFAULT_CONFIG_PATH = Path("models/serving_config.json")


def load_serving_config(path: str | Path | None = None) -> dict:
    path = Path(path or os.environ.get("SERVING_CONFIG", DEFAULT_CONFIG_PATH))
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration introuvable : {path}. Lancer d'abord "
            "`python scripts/run_all.py` (ou les etapes train, robustness, business)."
        )
    config = json.loads(path.read_text(encoding="utf-8"))
    threshold = float(config["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"seuil invalide dans {path} : {threshold}")
    return config | {"threshold": threshold}


def load_model(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Modele introuvable : {path}. Lancer d'abord `python scripts/run_all.py`.")
    return joblib.load(path)


def predict_one(model, threshold: float, features: dict) -> PredictionResponse:
    proba = float(model.predict_proba(pd.DataFrame([features]))[0, 1])
    return PredictionResponse(
        churn_probability=proba,
        churn_prediction=int(proba >= threshold),
        threshold=threshold,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_serving_config()
    app.state.model = load_model(os.environ.get("MODEL_PATH", config["model_path"]))
    app.state.threshold = config["threshold"]
    yield


app = FastAPI(title="Telco Churn Prediction API", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    return predict_one(app.state.model, app.state.threshold, customer.model_dump())
