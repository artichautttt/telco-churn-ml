import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shap_utils import (
    aggregate_by_feature,
    feature_origin,
    mean_abs_importance,
    tree_shap,
)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "tenure": rng.integers(0, 72, n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "Partner": rng.choice(["Yes", "No"], n),
    })
    y = ((X["Contract"] == "Month-to-month") & (X["tenure"] < 20)).astype(int)
    pre = ColumnTransformer([
        ("num", StandardScaler(), ["tenure"]),
        ("cat", OneHotEncoder(drop="if_binary"), ["Contract", "Partner"]),
    ])
    pipe = Pipeline([("preprocessor", pre),
                     ("model", XGBClassifier(n_estimators=20, max_depth=3, random_state=0))])
    return pipe.fit(X, y), X


def test_feature_origin_matches_transformed_width(fitted):
    pipe, _ = fitted
    origin = feature_origin(pipe.named_steps["preprocessor"])
    # tenure (1) + Contract (3 modalites) + Partner binaire (1 apres drop)
    assert origin == ["tenure", "Contract", "Contract", "Contract", "Partner"]


def test_tree_shap_is_additive(fitted):
    pipe, X = fitted
    values, Xt, names, explainer = tree_shap(pipe, X)
    margin = pipe.named_steps["model"].predict(Xt, output_margin=True)
    assert values.shape == Xt.shape == (len(X), len(names))
    assert np.allclose(values.sum(axis=1) + explainer.expected_value, margin, atol=1e-4)


def test_aggregation_preserves_row_sums(fitted):
    pipe, X = fitted
    values, *_ = tree_shap(pipe, X)
    origin = feature_origin(pipe.named_steps["preprocessor"])
    agg = aggregate_by_feature(values, origin)
    assert list(agg.columns) == ["tenure", "Contract", "Partner"]
    assert np.allclose(agg.sum(axis=1), values.sum(axis=1))
    imp = mean_abs_importance(values, origin)
    assert imp.index[-1] == "Partner"  # variable sans effet sur la cible


def test_aggregation_rejects_wrong_width():
    with pytest.raises(ValueError):
        aggregate_by_feature(np.zeros((2, 3)), ["a", "b"])
