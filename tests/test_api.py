import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import app, load_model

MODEL_PATH = ROOT / "models" / "best_model.pkl"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="models/best_model.pkl absent -- lancer `python -m src.train` d'abord",
)

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.7,
    "TotalCharges": 189.2,
}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    with TestClient(app) as c:  # le context manager declenche le lifespan (chargement du modele)
        yield c


def test_saved_model_loads_as_full_pipeline():
    pipeline = load_model(MODEL_PATH)
    assert list(pipeline.named_steps) == ["preprocessor", "model"]
    # Le preprocessing est bien inclus et deja ajuste : il accepte les features brutes.
    assert hasattr(pipeline.named_steps["preprocessor"], "transformers_")
    assert set(VALID_CUSTOMER) == set(pipeline.named_steps["preprocessor"].feature_names_in_)


def test_predict_valid_customer(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] == int(body["churn_probability"] >= 0.5)


@pytest.mark.parametrize("override", [
    {"tenure": -3},                     # hors plage
    {"Contract": "month-to-month"},     # categorie inconnue (serait ignoree par le OneHotEncoder)
    {"MonthlyCharges": "95.7"},         # mauvais type (pas de conversion implicite)
    {"MultipleLines": "No phone service"},  # incoherent avec PhoneService == "Yes"
])
def test_predict_invalid_input_returns_422(client, override):
    response = client.post("/predict", json=VALID_CUSTOMER | override)
    assert response.status_code == 422


def test_predict_missing_field_returns_422(client):
    payload = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    assert client.post("/predict", json=payload).status_code == 422
