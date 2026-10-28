import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.robustness import (
    bootstrap_ci,
    expected_calibration_error,
    group_metrics,
    rank_stability,
)


def test_bootstrap_ci_contains_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.4 + rng.uniform(0, 0.6, 400), 0, 1)
    a = bootstrap_ci(y, p, n_boot=200)
    b = bootstrap_ci(y, p, n_boot=200)
    assert a == b
    for k in ("auc", "recall"):
        lo, hi = a[k]["ci95"]
        assert lo <= a[k]["point"] <= hi


def test_rank_stability_identical_rankings():
    s = pd.Series(np.arange(12, 0, -1), index=[f"f{i}" for i in range(12)], dtype=float)
    r = rank_stability([s, s.copy(), s.copy()])
    assert r["spearman_mean"] == pytest.approx(1.0)
    assert r["top_overlap_min"] == 1.0


def test_rank_stability_detects_reversal():
    s = pd.Series(np.arange(10, 0, -1), index=[f"f{i}" for i in range(10)], dtype=float)
    r = rank_stability([s, pd.Series(s.values[::-1], index=s.index)])
    assert r["spearman_mean"] == pytest.approx(-1.0)


def test_ece_perfect_and_biased():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.5)) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.full(100, 0.9)) == pytest.approx(0.4)


def test_group_metrics_counts():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.7])
    g = np.array(["A", "A", "A", "B", "B", "B"])
    res = group_metrics(y, p, g).set_index("group")
    assert res.loc["A", "n"] == 3 and res.loc["A", "recall"] == 1.0
    assert res.loc["B", "recall"] == 0.5
    assert res["n"].sum() == len(y)
