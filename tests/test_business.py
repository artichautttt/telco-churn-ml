import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.business import Costs, capture_at, gains_curve, optimal_threshold, total_cost


def test_total_cost_by_cell():
    c = Costs(action_cost=10, churn_loss=100, success_rate=0.5)
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.9, 0.1, 0.1])  # TP, FP, FN, TN au seuil 0,5
    assert total_cost(y, p, 0.5, c) == pytest.approx((10 + 50) + 10 + 100 + 0)


def test_theoretical_threshold():
    assert Costs(50, 500, 0.5).theoretical_threshold() == pytest.approx(0.2)


def test_optimal_threshold_matches_theory_on_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)  # probabilites parfaitement calibrees
    t, curve = optimal_threshold(y, p, Costs(50, 500, 0.5))
    assert abs(t - 0.2) <= 0.03
    assert len(curve) > 0


def test_gains_and_capture():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.arange(10, 0, -1)  # les deux churners ont les meilleurs scores
    frac, captured = gains_curve(y, s)
    assert captured[-1] == 1.0 and frac[0] == 0
    assert capture_at(y, s, 0.1) == 0.5
    assert capture_at(y, s, 0.2) == 1.0
