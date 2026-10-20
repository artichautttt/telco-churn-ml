"""
api.py -- API FastAPI de serving du modele de churn.

Charge le pipeline complet sauvegarde par train.py (ColumnTransformer +
modele, un seul fichier joblib) : le pretraitement applique en serving est
donc exactement celui de l'entrainement.

Lancement : uvicorn src.api:app --reload
Chemin du modele configurable via la variable d'environnement MODEL_PATH
(defaut : models/best_model.pkl).
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from src.schemas import CustomerFeatures, PredictionResponse

DEFAULT_MODEL_PATH = Path("models/best_model.pkl")


def load_model(path: str | Path | None = None):
    path = Path(path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.exists():
        raise FileNotFoundError(
            f"Modele introuvable : {path}. Lancer d'abord `python -m src.train`."
        )
    return joblib.load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(title="Telco Churn Prediction API", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    X = pd.DataFrame([customer.model_dump()])
    model = app.state.model
    return PredictionResponse(
        churn_probability=float(model.predict_proba(X)[0, 1]),
        churn_prediction=int(model.predict(X)[0]),
    )
