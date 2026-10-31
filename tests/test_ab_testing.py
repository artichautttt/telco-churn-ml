import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ab_testing import (
    _z_pvalues,
    aa_false_positive_rate,
    analyze_groups,
    assign_groups,
    balance_table,
    campaign_cost,
    expected_net_benefit,
    minimum_detectable_effect,
    required_sample_size,
    simulate_treatment_effect,
    simulated_power,
    srm_test,
    target_mask,
    theoretical_power,
    two_proportion_test,
)

# --- Taille d'echantillon ----------------------------------------------------

def test_required_sample_size_matches_closed_form():
    # 50 % -> 40 % (reduction relative 20 %), alpha 5 %, puissance 80 %.
    # Formule fermee (arcsinus) : n = 2 * ((z_{1-a/2} + z_{1-b}) / h)^2.
    h = 2 * math.asin(math.sqrt(0.5)) - 2 * math.asin(math.sqrt(0.4))
    closed = 2 * ((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) / h) ** 2
    n = required_sample_size(0.5, 0.2, alpha=0.05, power=0.8)
    assert n == math.ceil(closed)
    assert 380 <= n <= 390  # ordre de grandeur des tables usuelles (~385-388)


def test_required_sample_size_grows_when_effect_shrinks():
    sizes = [required_sample_size(0.45, r) for r in (0.5, 0.3, 0.2, 0.1)]
    assert sizes == sorted(sizes)
    assert required_sample_size(0.45, 0.2, power=0.9) > required_sample_size(0.45, 0.2, power=0.8)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
def test_required_sample_size_rejects_invalid_effect(bad):
    with pytest.raises(ValueError):
        required_sample_size(0.45, bad)


def test_minimum_detectable_effect_is_inverse_of_sample_size():
    mde = minimum_detectable_effect(0.45, 400)
    assert 0 < mde["relative_reduction"] < 1
    assert mde["absolute"] < 0
    # n requis pour cet effet = n disponible (a l'arrondi pres)
    assert abs(required_sample_size(0.45, mde["relative_reduction"]) - 400) <= 1
    assert theoretical_power(0.45, mde["relative_reduction"], 400) == pytest.approx(0.8, abs=1e-3)


def test_theoretical_power_without_effect_is_alpha():
    assert theoretical_power(0.45, 0.0, 300, alpha=0.05) == 0.05


# --- Test de deux proportions ------------------------------------------------

def test_two_proportion_test_by_hand():
    # controle 45/100, traitement 30/100
    res = two_proportion_test(45, 100, 30, 100)
    pooled = 75 / 200
    z = (0.30 - 0.45) / math.sqrt(pooled * (1 - pooled) * (2 / 100))
    assert res.z == pytest.approx(z)
    assert res.p_value == pytest.approx(2 * stats.norm.sf(abs(z)))
    assert res.absolute_diff == pytest.approx(-0.15)
    assert res.relative_diff == pytest.approx(-0.15 / 0.45)
    se = math.sqrt(0.3 * 0.7 / 100 + 0.45 * 0.55 / 100)
    assert res.ci_low == pytest.approx(-0.15 - stats.norm.ppf(0.975) * se)
    assert res.ci_high == pytest.approx(-0.15 + stats.norm.ppf(0.975) * se)
    assert res.significant == (res.p_value < 0.05)


def test_two_proportion_test_matches_chi_square():
    res = two_proportion_test(120, 370, 90, 366)
    table = [[90, 366 - 90], [120, 370 - 120]]
    chi2 = stats.chi2_contingency(table, correction=False)
    assert res.z ** 2 == pytest.approx(chi2.statistic)
    assert res.p_value == pytest.approx(chi2.pvalue)


def test_vectorized_z_matches_statsmodels():
    churn_t, n_t, churn_c, n_c = [90, 30, 0], [366, 100, 50], [120, 45, 0], [370, 100, 50]
    z, p = _z_pvalues(churn_t, n_t, churn_c, n_c)
    for i in range(2):
        z_ref, p_ref = proportions_ztest([churn_t[i], churn_c[i]], [n_t[i], n_c[i]])
        assert z[i] == pytest.approx(z_ref)
        assert p[i] == pytest.approx(p_ref)
    assert z[2] == 0 and p[2] == 1  # aucun churn : pas de difference, pas d'erreur


def test_analyze_groups_counts():
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    treated = np.array([True, True, True, True, False, False, False, False])
    res = analyze_groups(y, treated)
    assert (res.n_treatment, res.churn_treatment, res.n_control, res.churn_control) == (4, 3, 4, 1)


# --- Randomisation -----------------------------------------------------------

def test_assign_groups_reproducible_and_balanced():
    a, b = assign_groups(10_000, seed=42), assign_groups(10_000, seed=42)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, assign_groups(10_000, seed=43))
    assert 0.48 < a.mean() < 0.52


def test_srm_test():
    assert srm_test(500, 500)["p_value"] == pytest.approx(1.0)
    assert srm_test(400, 600)["p_value"] < 1e-9


def test_balance_table_flags_imbalance():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0] * 50, "c": ["a", "b"] * 100})
    treated = np.array([True, False] * 100)  # c = "a" seulement chez les traites
    table = balance_table(df, treated, numeric=["x"], categorical=["c"])
    assert set(table["variable"]) == {"x", "c"}
    assert table.loc[table["variable"] == "c", "p_value"].iloc[0] < 1e-9
    row_a = table[(table["variable"] == "c") & (table["level"] == "a")].iloc[0]
    assert (row_a["treatment"], row_a["control"]) == (1.0, 0.0)


# --- Effet simule ------------------------------------------------------------

def test_simulated_effect_zero_changes_nothing():
    y = np.array([1, 0, 1, 1, 0, 1])
    treated = np.array([True, True, True, False, False, False])
    assert np.array_equal(simulate_treatment_effect(y, treated, 0.0, seed=0), y)


def test_simulated_effect_full_removes_treated_churn_only():
    rng = np.random.default_rng(1)
    y = (rng.random(1_000) < 0.45).astype(int)
    treated = assign_groups(1_000, seed=2)
    y_obs = simulate_treatment_effect(y, treated, 1.0, seed=3)
    assert y_obs[treated].sum() == 0
    assert np.array_equal(y_obs[~treated], y[~treated])  # controle : vrais labels


def test_simulated_effect_reduces_by_expected_fraction():
    y = np.ones(20_000, dtype=int)
    treated = np.ones(20_000, dtype=bool)
    y_obs = simulate_treatment_effect(y, treated, 0.3, seed=4)
    assert 1 - y_obs.mean() == pytest.approx(0.3, abs=0.01)
    assert np.array_equal(y_obs, simulate_treatment_effect(y, treated, 0.3, seed=4))


def test_simulated_effect_rejects_invalid_reduction():
    with pytest.raises(ValueError):
        simulate_treatment_effect([1, 0], [True, False], 1.5, seed=0)


# --- Test A/A et puissance simulee -------------------------------------------

def test_aa_false_positive_rate_close_to_alpha():
    y = (np.random.default_rng(5).random(700) < 0.45).astype(int)
    aa = aa_false_positive_rate(y, n_sims=2_000, seed=6, alpha=0.05)
    assert aa["n_sims"] == 2_000
    assert 0.035 <= aa["rate"] <= 0.065
    assert aa["ci_low"] <= aa["rate"] <= aa["ci_high"]
    assert aa == aa_false_positive_rate(y, n_sims=2_000, seed=6, alpha=0.05)


def test_aa_rate_follows_alpha():
    y = (np.random.default_rng(7).random(700) < 0.45).astype(int)
    assert aa_false_positive_rate(y, 2_000, seed=8, alpha=0.01)["rate"] < 0.025


def test_simulated_power_extremes():
    y = (np.random.default_rng(9).random(700) < 0.45).astype(int)
    assert simulated_power(y, 0.9, n_sims=500, seed=10) > 0.99
    # sans effet : seules les fausses baisses comptent, ~ alpha / 2
    assert simulated_power(y, 0.0, n_sims=2_000, seed=11) < 0.05


# --- Cout ---------------------------------------------------------------------

def test_target_mask_uses_greater_or_equal():
    assert target_mask([0.1, 0.18, 0.5], 0.18).tolist() == [False, True, True]


def test_cost_and_break_even():
    assert campaign_cost(368, 50.0) == 18_400.0
    out = expected_net_benefit(100, p_control=0.4, relative_reduction=0.25, action_cost=50.0,
                               churn_loss=500.0)
    assert out["break_even_relative_reduction"] == pytest.approx(0.25)
    assert out["avoided_churners"] == pytest.approx(10.0)
    assert out["net_benefit"] == pytest.approx(0.0)
