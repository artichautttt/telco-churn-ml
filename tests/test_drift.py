import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import drift_report
from src.drift import (
    EPS,
    DriftSimulator,
    Thresholds,
    alert_level,
    bh_adjust,
    chi2_test,
    choose_psi_threshold,
    drift_columns,
    evaluate_draw,
    feature_drift,
    inject_concept_drift,
    inject_contract_mix,
    inject_monthly_charges_increase,
    inject_tenure_shift,
    ks_test,
    numeric_bin_edges,
    overall_level,
    performance_at_threshold,
    prediction_drift,
    psi_categorical,
    psi_numeric,
    random_split,
    rate_with_se,
)


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(18, 120, n).round(2)
    return pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly,
        "SeniorCitizen": rng.integers(0, 2, n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]),
        "gender": rng.choice(["Female", "Male"], n),
    })


# --- PSI ---------------------------------------------------------------------

def test_psi_zero_on_identical_distributions():
    x = np.random.default_rng(0).normal(size=5_000)
    assert psi_numeric(x, x) == pytest.approx(0.0, abs=1e-12)
    cats = ["a", "b", "b", "c"] * 100
    assert psi_categorical(cats, cats) == pytest.approx(0.0, abs=1e-12)


def test_psi_numeric_known_value_by_hand():
    ref = np.arange(100)                        # mediane 49,5 : deux classes de 50 %
    cur = np.r_[np.zeros(25), np.full(75, 99)]  # 25 % sous la mediane, 75 % au-dessus
    expected = (0.25 - 0.5) * math.log(0.25 / 0.5) + (0.75 - 0.5) * math.log(0.75 / 0.5)
    assert psi_numeric(ref, cur, n_bins=2) == pytest.approx(expected)
    assert expected == pytest.approx(0.2747, abs=1e-4)


def test_psi_categorical_known_value_by_hand():
    ref = ["a"] * 50 + ["b"] * 50
    cur = ["a"] * 20 + ["b"] * 80
    expected = (0.2 - 0.5) * math.log(0.2 / 0.5) + (0.8 - 0.5) * math.log(0.8 / 0.5)
    assert psi_categorical(ref, cur) == pytest.approx(expected)


def test_psi_bins_fixed_on_reference_and_ties_merged():
    ref = np.r_[np.zeros(80), np.arange(1, 21)]  # 80 % de zeros : quantiles en double
    edges = numeric_bin_edges(ref, n_bins=10)
    assert len(edges) == len(np.unique(edges)) < 9
    # les classes ne bougent pas avec le courant
    assert np.array_equal(edges, numeric_bin_edges(ref, n_bins=10))


def test_psi_empty_bins_stay_finite():
    ref = np.arange(1_000)
    cur = np.full(200, 5.0)  # tout le courant dans la premiere classe, les autres vides
    value = psi_numeric(ref, cur)
    assert math.isfinite(value) and value > 1
    # classes vides remplacees par EPS : valeur exacte
    ref_p = np.full(10, 0.1)
    cur_p = np.r_[1.0, np.full(9, EPS)]
    assert value == pytest.approx(float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p))))


def test_unknown_category_counts_in_psi_and_chi2():
    ref = ["a"] * 60 + ["b"] * 40
    cur = ["a"] * 50 + ["b"] * 30 + ["z"] * 20   # z absent de la reference
    value = psi_categorical(ref, cur)
    expected = ((0.5 - 0.6) * math.log(0.5 / 0.6) + (0.3 - 0.4) * math.log(0.3 / 0.4)
                + (0.2 - EPS) * math.log(0.2 / EPS))
    assert value == pytest.approx(expected)
    stat, p = chi2_test(ref, cur)
    ref_stat = stats.chi2_contingency([[60, 40, 0], [50, 30, 20]], correction=False)
    assert stat == pytest.approx(ref_stat.statistic) and p == pytest.approx(ref_stat.pvalue)


def test_chi2_single_category_is_not_an_error():
    assert chi2_test(["a"] * 10, ["a"] * 5) == (0.0, 1.0)


# --- Tests statistiques et correction ----------------------------------------

def test_ks_matches_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=800), rng.normal(0.2, 1, size=300)
    ref = stats.ks_2samp(a, b)
    assert ks_test(a, b) == pytest.approx((ref.statistic, ref.pvalue))


def test_bh_matches_statsmodels():
    p = [0.001, 0.02, 0.03, 0.2, 0.5]
    assert np.allclose(bh_adjust(p), multipletests(p, method="fdr_bh")[1])
    assert bh_adjust([]).size == 0


# --- Niveaux d'alerte --------------------------------------------------------

@pytest.mark.parametrize("psi,p,level", [
    (0.30, 0.001, "alerte"),
    (0.30, 0.20, "surveillance"),     # gros ecart mais compatible avec le hasard
    (0.12, 0.20, "surveillance"),
    (0.02, 0.001, "surveillance"),    # significatif mais minuscule
    (0.02, 0.20, "aucun"),
])
def test_alert_level_convention(psi, p, level):
    assert alert_level(psi, p) == level


def test_alert_level_uses_given_thresholds():
    assert alert_level(0.30, 0.001, Thresholds(psi_watch=0.2, psi_alert=0.4)) == "surveillance"
    assert overall_level(["aucun", "alerte", "surveillance"]) == "alerte"
    assert overall_level([]) == "aucun"


def test_feature_drift_detects_injected_shift_only():
    ref, cur = _frame(4_000, 2), _frame(1_000, 3)
    numeric, categorical = drift_columns(ref)
    assert numeric == ["tenure", "MonthlyCharges", "TotalCharges"]
    assert "SeniorCitizen" in categorical
    calm = feature_drift(ref, cur, numeric, categorical)
    assert set(calm["level"]) <= {"aucun", "surveillance"}
    shifted = feature_drift(ref, inject_monthly_charges_increase(cur, 0.5), numeric, categorical)
    row = shifted.set_index("variable").loc["MonthlyCharges"]
    assert row["level"] == "alerte" and row["psi"] > 0.25
    assert np.allclose(shifted["p_adjusted"], multipletests(shifted["p_value"], method="fdr_bh")[1])


def test_prediction_drift_share_above_threshold():
    out = prediction_drift([0.1, 0.2, 0.3, 0.9], [0.5, 0.6, 0.7, 0.05], threshold=0.18)
    assert out["share_above_ref"] == 0.75 and out["share_above_cur"] == 0.75
    assert out["share_above_diff"] == 0.0 and out["level"] in {"aucun", "surveillance", "alerte"}


# --- Injections et reproductibilite ------------------------------------------

def test_injections_are_reproducible_and_targeted():
    df = _frame(2_000, 4)
    a = inject_contract_mix(df, 0.80, seed=7)
    assert a.equals(inject_contract_mix(df, 0.80, seed=7))
    assert not a.equals(inject_contract_mix(df, 0.80, seed=8))
    assert (a["Contract"] == "Month-to-month").mean() == pytest.approx(0.80, abs=1e-3)
    assert a.drop(columns="Contract").equals(df.drop(columns="Contract"))  # seule Contract change


def test_tenure_shift_keeps_total_charges_consistent():
    df = _frame(500, 5)
    out = inject_tenure_shift(df, 6)
    assert (out["tenure"] == np.maximum(df["tenure"] - 6, 0)).all()
    pos = df["tenure"] > 0
    assert np.allclose(out.loc[pos, "TotalCharges"], df.loc[pos, "TotalCharges"] * out.loc[pos, "tenure"]
                       / df.loc[pos, "tenure"])
    assert out.drop(columns=["tenure", "TotalCharges"]).equals(df.drop(columns=["tenure", "TotalCharges"]))


def test_concept_drift_changes_labels_not_features():
    df = _frame(3_000, 6)
    y = np.zeros(len(df), dtype=int)
    y_new = inject_concept_drift(df, y, 0.3, tenure_cut=36, seed=9)
    assert np.array_equal(y_new, inject_concept_drift(df, y, 0.3, tenure_cut=36, seed=9))
    assert y_new[df["tenure"].to_numpy() <= 36].sum() == 0
    assert y_new[df["tenure"].to_numpy() > 36].mean() == pytest.approx(0.3, abs=0.04)


def test_random_split_reproducible_and_disjoint():
    ref, cur = random_split(100, 20, seed=1)
    ref2, cur2 = random_split(100, 20, seed=1)
    assert np.array_equal(ref, ref2) and np.array_equal(cur, cur2)
    assert len(cur) == 20 and len(set(ref) | set(cur)) == 100 and not set(ref) & set(cur)


def test_performance_and_rate_with_se():
    perf = performance_at_threshold([1, 1, 0, 0], [0.9, 0.1, 0.5, 0.05], 0.18)
    assert perf["recall"] == 0.5 and perf["precision"] == 0.5
    r = rate_with_se(50, 1_000)
    assert r["rate"] == 0.05 and r["se"] == pytest.approx(math.sqrt(0.05 * 0.95 / 1_000))


# --- Rapport ----------------------------------------------------------------

def _report():
    ref, cur = _frame(1_000, 10), _frame(400, 11)
    rng = np.random.default_rng(12)
    return drift_report.build_report(ref, cur, rng.random(1_000), rng.random(400), threshold=0.18,
                                     thresholds=Thresholds(), reference_name="ref", current_name="cur",
                                     model_path="models/x.pkl")


def test_report_json_schema(tmp_path):
    report = _report()
    assert tuple(report) == drift_report.REPORT_KEYS
    json_path, md_path = drift_report.write_report(report, tmp_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(loaded["summary"]) == {"n_features", "n_surveillance", "n_alerte", "flagged", "overall_level"}
    assert loaded["summary"]["n_features"] == len(loaded["features"]) == 6
    assert set(loaded["features"][0]) == {"variable", "type", "test", "psi", "statistic", "p_value",
                                          "p_adjusted", "level"}
    assert set(loaded["prediction"]) >= {"psi", "p_value", "share_above_ref", "share_above_cur", "level"}
    assert loaded["thresholds"] == {"psi_watch": 0.1, "psi_alert": 0.25, "alpha": 0.05}
    assert "derive reelle" in loaded["disclaimer"]
    assert md_path.read_text(encoding="utf-8").startswith("# Rapport de derive")


def test_read_current_csv_validates_path_and_columns(tmp_path):
    cols = list(_frame(5, 0).columns)
    with pytest.raises(drift_report.DriftInputError, match="introuvable"):
        drift_report.read_current_csv(tmp_path / "absent.csv", cols)
    bad = tmp_path / "bad.csv"
    _frame(5, 0).drop(columns=["Contract"]).to_csv(bad, index=False)
    with pytest.raises(drift_report.DriftInputError, match="colonnes manquantes.*Contract"):
        drift_report.read_current_csv(bad, cols)
    raw = _frame(5, 0).assign(TotalCharges=" ", customerID="x", Churn="No")
    good = tmp_path / "good.csv"
    raw.to_csv(good, index=False)
    out = drift_report.read_current_csv(good, cols)
    assert list(out.columns) == cols and (out["TotalCharges"] == 0).all()


def test_mlflow_failure_never_raises(monkeypatch):
    import mlflow

    def boom(*args, **kwargs):
        raise RuntimeError("mlflow indisponible")

    monkeypatch.setattr(mlflow, "set_tracking_uri", boom)
    assert drift_report.log_to_mlflow(_report()) is False


def test_mlflow_logs_to_given_uri(tmp_path):
    uri = (tmp_path / "mlruns").as_uri()
    assert drift_report.log_to_mlflow(_report(), tracking_uri=uri) is True
    assert any((tmp_path / "mlruns").iterdir())


def test_choose_psi_threshold_keeps_convention_or_raises_it():
    null = np.linspace(0.0, 0.05, 101)            # tous sous la convention
    assert choose_psi_threshold(null, 0.01, 0.10) == 0.10
    null = np.r_[np.zeros(90), np.full(10, 0.3)]  # 10 % de fausses alertes au-dessus de 0,25
    assert choose_psi_threshold(null, 0.05, 0.25) == pytest.approx(0.3)


def test_evaluate_draw_levels():
    ref, cur = _frame(2_000, 13), _frame(500, 14)
    rng = np.random.default_rng(15)
    out = evaluate_draw(ref, inject_monthly_charges_increase(cur, 0.5), rng.random(2_000), rng.random(500), 0.18)
    assert out["features_level"] == "alerte" and out["overall_level"] == "alerte"
    assert set(out["features"]["variable"]) == set(ref.columns)


def test_drift_simulator_matches_feature_and_prediction_drift():
    pool = _frame(3_000, 16)
    probs = np.random.default_rng(17).random(3_000)
    shifted = inject_contract_mix(inject_monthly_charges_increase(pool, 0.05), 0.75, seed=18)
    shifted.loc[shifted.index[:30], "gender"] = "Unknown"  # modalite inconnue de la reference
    p_shifted = np.clip(probs * 1.1, 0, 1)
    sim = DriftSimulator(pool, probs, threshold=0.18)
    ref_idx, cur_idx = random_split(3_000, 700, seed=19)
    fast = sim.evaluate(ref_idx, sim.encode(shifted), p_shifted, cur_idx)
    slow = feature_drift(pool.iloc[ref_idx], shifted.iloc[cur_idx]).set_index("variable").loc[sim.variables]
    assert np.allclose(fast["psi"], slow["psi"]) and np.allclose(fast["p_value"], slow["p_value"])
    assert np.allclose(fast["p_adjusted"], slow["p_adjusted"]) and fast["levels"] == list(slow["level"])
    assert fast["prediction"] == prediction_drift(probs[ref_idx], p_shifted[cur_idx], 0.18)
