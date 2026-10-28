import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.interactions import (
    InteractionTerms,
    aggregate_interactions,
    rank_pairs,
    slope_by_group,
    stratified_sample,
    without_feature,
)


def test_aggregate_interactions_sums_one_hot_blocks():
    rng = np.random.default_rng(0)
    inter = rng.normal(size=(4, 3, 3))
    agg, feats = aggregate_interactions(inter, ["a", "b", "b"])
    assert feats == ["a", "b"]
    assert agg.shape == (4, 2, 2)
    assert np.allclose(agg[:, 0, 1], inter[:, 0, 1] + inter[:, 0, 2])
    assert np.allclose(agg[:, 1, 1], inter[:, 1:, 1:].sum(axis=(1, 2)))
    assert np.allclose(agg.sum(axis=(1, 2)), inter.sum(axis=(1, 2)))


def test_rank_pairs_orders_by_strength():
    agg = np.zeros((2, 3, 3))
    agg[:, 0, 2] = agg[:, 2, 0] = 1.0   # a-c forte
    agg[:, 0, 1] = agg[:, 1, 0] = 0.1   # a-b faible
    ranking = rank_pairs(agg, ["a", "b", "c"])
    assert tuple(ranking.iloc[0][["feature_a", "feature_b"]]) == ("a", "c")
    assert ranking.iloc[0]["strength"] == pytest.approx(2.0)
    assert len(ranking) == 3


def test_slope_by_group():
    x = np.array([0, 1, 2, 0, 1, 2], dtype=float)
    y = np.array([0, 2, 4, 0, -1, -2], dtype=float)
    g = np.array(["A"] * 3 + ["B"] * 3)
    slopes = slope_by_group(x, y, g)
    assert slopes["A"] == pytest.approx(2.0)
    assert slopes["B"] == pytest.approx(-1.0)


def test_interaction_terms_adds_products_for_non_reference_levels():
    X = pd.DataFrame({
        "tenure": [1, 2, 3], "Contract": ["Month-to-month", "One year", "Two year"],
        "MonthlyCharges": [10.0, 20.0, 30.0], "InternetService": ["DSL", "Fiber optic", "No"],
    })
    Z = InteractionTerms().fit(X).transform(X)
    assert "tenure_x_Contract=One year" in Z and "tenure_x_Contract=Month-to-month" not in Z
    assert list(Z["tenure_x_Contract=Two year"]) == [0, 0, 3]
    assert list(Z["MonthlyCharges_x_InternetService=Fiber optic"]) == [0, 20.0, 0]


def test_mechanical_pairs_flagged_and_filtered():
    feats = ["tenure", "TotalCharges", "Contract"]
    agg = np.zeros((1, 3, 3))
    agg[:, 0, 1] = agg[:, 1, 0] = 1.0   # tenure x TotalCharges (mecanique)
    agg[:, 0, 2] = agg[:, 2, 0] = 0.5   # tenure x Contract
    ranking = rank_pairs(agg, feats)
    assert bool(ranking.iloc[0]["mechanical"]) is True
    assert not ranking[ranking["feature_b"] == "Contract"]["mechanical"].any()
    informative = without_feature(ranking)
    assert "TotalCharges" not in set(informative["feature_a"]) | set(informative["feature_b"])
    assert tuple(informative.iloc[0][["feature_a", "feature_b"]]) == ("tenure", "Contract")


def test_stratified_sample_size_and_rate():
    X = pd.DataFrame({"a": range(2000)})
    y = pd.Series([1] * 500 + [0] * 1500)
    Xs, ys = stratified_sample(X, y, 1000)
    assert len(Xs) == 1000 and ys.mean() == 0.25
    Xa, _ = stratified_sample(X, y, 5000)
    assert len(Xa) == 2000
