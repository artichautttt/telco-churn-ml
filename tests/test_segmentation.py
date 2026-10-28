import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segmentation import (
    choose_k,
    encode_for_kmeans,
    fit_kprototypes,
    gower_distance,
    profile_segments,
    shap_rank_by_segment,
    split_types,
)


def toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat([0, 1], n // 2)
    X = pd.DataFrame({
        "tenure": np.where(g == 0, rng.integers(0, 10, n), rng.integers(50, 72, n)),
        "MonthlyCharges": np.where(g == 0, 90.0, 30.0) + rng.normal(0, 2, n),
        "TotalCharges": rng.uniform(0, 5000, n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Contract": np.where(g == 0, "Month-to-month", "Two year"),
        "InternetService": np.where(g == 0, "Fiber optic", "DSL"),
    })
    y = pd.Series(np.where(g == 0, 1, 0))
    return X, y, g


def test_split_types_treats_senior_citizen_as_categorical():
    num, cat = split_types(toy()[0])
    assert num == ["tenure", "MonthlyCharges", "TotalCharges"]
    assert "SeniorCitizen" in cat


def test_gower_distance_properties():
    num = np.array([[0.0], [1.0], [2.0]])
    cat = np.array([["a"], ["a"], ["b"]])
    d = gower_distance(num, cat)
    assert np.allclose(np.diag(d), 0)
    assert np.allclose(d, d.T)
    assert d[0, 2] == pytest.approx((1.0 + 1.0) / 2)  # ecart max + modalite differente
    assert d[0, 1] == pytest.approx((0.5 + 0.0) / 2)


def test_choose_k_restricted_to_business_range():
    assert choose_k({2: 0.9, 3: 0.3, 4: 0.5, 5: 0.4, 6: 0.8}) == 4
    with pytest.raises(ValueError):
        choose_k({2: 0.5})


def test_kprototypes_recovers_two_obvious_groups():
    X, _, g = toy()
    labels, cost = fit_kprototypes(X, 2, n_init=1)
    assert cost > 0
    agree = max((labels == g).mean(), (labels != g).mean())
    assert agree == 1.0


def test_encode_for_kmeans_shape():
    X = toy()[0]
    Z = encode_for_kmeans(X)
    assert Z.shape[0] == len(X)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-9)


def test_profile_segments_sizes_and_rates():
    X, y, g = toy()
    prof = profile_segments(X, y, g)
    assert prof["n_clients"].sum() == len(X)
    assert prof["share_of_churners"].sum() == pytest.approx(1.0)
    assert prof.set_index("segment").loc[0, "churn_rate"] == 1.0
    assert "Month-to-month" in prof.set_index("segment").loc[0, "distinctive"]


def test_shap_rank_by_segment_detects_different_drivers():
    shap_agg = pd.DataFrame({"a": [5, 5, 0.1, 0.1], "b": [0.1, 0.1, 5, 5], "c": [1, 1, 1, 1]})
    res = shap_rank_by_segment(shap_agg, np.array([0, 0, 1, 1]), top=1).set_index("segment")
    assert res.loc[0, "top_factors"] == "a"
    assert res.loc[1, "top_factors"] == "b"
