import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import app, load_model, load_serving_config, predict_one

CONFIG_PATH = ROOT / "models" / "serving_config.json"
BEST_MODEL_PATH = ROOT / "models" / "best_model.pkl"
SPLIT_PATH = ROOT / "models" / "test_split.pkl"

requires_artifacts = pytest.mark.skipif(
    not (CONFIG_PATH.exists() and BEST_MODEL_PATH.exists() and SPLIT_PATH.exists()),
    reason="modeles absents -- lancer `python scripts/run_all.py` d'abord",
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


# --- Tests sans artefacts : logique de decision et configuration -----------

class ConstantModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.array([[1 - self.p, self.p]] * len(X))


@pytest.mark.parametrize("proba, expected", [(0.17, 0), (0.18, 1), (0.40, 1)])
def test_decision_uses_threshold_not_one_half(proba, expected):
    r = predict_one(ConstantModel(proba), 0.18, VALID_CUSTOMER)
    assert r.churn_prediction == expected
    assert r.threshold == 0.18
    assert r.churn_probability == pytest.approx(proba)


def test_serving_config_rejects_invalid_threshold(tmp_path):
    cfg = tmp_path / "serving.json"
    cfg.write_text(json.dumps({"model_path": "x.pkl", "threshold": 1.5}))
    with pytest.raises(ValueError):
        load_serving_config(cfg)


def test_serving_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_serving_config(tmp_path / "absent.json")


# --- Tests sur les artefacts produits par run_all.py -----------------------

@pytest.fixture(scope="module")
def config():
    return load_serving_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    with TestClient(app) as c:  # le context manager declenche le lifespan (chargement du modele)
        yield c


@requires_artifacts
def test_served_model_is_calibrated_full_pipeline(config):
    model = load_model(ROOT / config["model_path"])
    assert type(model).__name__ == "CalibratedClassifierCV"
    inner = model.calibrated_classifiers_[0].estimator
    assert list(inner.named_steps) == ["preprocessor", "model"]
    assert set(VALID_CUSTOMER) == set(inner.named_steps["preprocessor"].feature_names_in_)


@requires_artifacts
def test_threshold_comes_from_cost_matrix(config):
    assert config["threshold"] != 0.5
    assert "costs_hypotheses" in config


@requires_artifacts
def test_recalibration_preserves_ranking(config):
    """La recalibration (Platt, monotone) ne doit pas changer le classement."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    split = joblib.load(SPLIT_PATH)
    X, y = split["X_test"], split["y_test"]
    p_cal = load_model(ROOT / config["model_path"]).predict_proba(X)[:, 1]
    p_raw = load_model(BEST_MODEL_PATH).predict_proba(X)[:, 1]
    assert roc_auc_score(y, p_cal) == pytest.approx(roc_auc_score(y, p_raw), abs=1e-12)
    assert spearmanr(p_cal, p_raw).statistic == pytest.approx(1.0)


@requires_artifacts
def test_predict_valid_customer(client, config):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"churn_probability", "churn_prediction", "threshold"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["threshold"] == config["threshold"]
    assert body["churn_prediction"] == int(body["churn_probability"] >= body["threshold"])


@requires_artifacts
@pytest.mark.parametrize("override", [
    {"tenure": -3},                     # hors plage
    {"Contract": "month-to-month"},     # categorie inconnue (serait ignoree par le OneHotEncoder)
    {"MonthlyCharges": "95.7"},         # mauvais type (pas de conversion implicite)
    {"MultipleLines": "No phone service"},  # incoherent avec PhoneService == "Yes"
])
def test_predict_invalid_input_returns_422(client, override):
    response = client.post("/predict", json=VALID_CUSTOMER | override)
    assert response.status_code == 422


@requires_artifacts
def test_predict_missing_field_returns_422(client):
    payload = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    assert client.post("/predict", json=payload).status_code == 422
