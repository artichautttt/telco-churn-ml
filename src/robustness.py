"""
robustness.py -- robustesse des resultats (etape 5).

- IC a 95 % par bootstrap (1 000 tirages, percentile) de l'AUC et du rappel
  churn sur le test set, pour le modele servi et pour XGBoost ;
- stabilite du top 10 SHAP (XGBoost) : sur les 5 folds de la CV (modele
  reentraine sur chaque fold, importance mesuree sur le meme test set) et sur
  5 seeds (nouveau decoupage train/test et nouvelle seed du modele) ;
  correlation de rang de Spearman deux a deux sur l'union des top 10 ;
- calibration du modele servi : courbe, score de Brier, ECE ; recalibration
  isotonique et Platt (sigmoid) par CalibratedClassifierCV(cv=5,
  ensemble=False) sur le train : un seul modele ajuste sur tout le train, suivi
  d'une fonction monotone apprise hors-fold (Platt : strictement croissante,
  donc classement et AUC identiques au modele servi) ; methode choisie sur
  le Brier hors-fold du train (le test set ne sert qu'a l'evaluation
  finale) ; recalibration retenue seulement si elle baisse le Brier ;
- performance par segment (etape 3) et par type de contrat.

Usage : python -m src.robustness
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src.preprocessing import clean_data, load_data, split_X_y
from src.shap_utils import feature_origin, mean_abs_importance, tree_shap
from src.train import RANDOM_STATE, TEST_SIZE, compute_scale_pos_weight

N_BOOT = 1000
SEEDS = (0, 1, 2, 3, 4)
TOP_N = 10


def bootstrap_ci(y_true, y_proba, threshold=0.5, n_boot=N_BOOT, seed=RANDOM_STATE, alpha=0.05):
    """IC percentile de l'AUC et du rappel (tirages avec remise, stratifies
    implicitement non : un tirage sans positif ou sans negatif est ignore)."""
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    rng = np.random.default_rng(seed)
    aucs, recalls = [], []
    n = len(y_true)
    while len(aucs) < n_boot:
        i = rng.integers(0, n, n)
        if y_true[i].min() == y_true[i].max():
            continue
        aucs.append(roc_auc_score(y_true[i], y_proba[i]))
        recalls.append(recall_score(y_true[i], y_pred[i]))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {
        "auc": {"point": float(roc_auc_score(y_true, y_proba)),
                "ci95": [float(v) for v in np.percentile(aucs, q)]},
        "recall": {"point": float(recall_score(y_true, y_pred)),
                   "ci95": [float(v) for v in np.percentile(recalls, q)]},
        "n_boot": n_boot,
    }


def rank_stability(importances: list[pd.Series], top_n: int = TOP_N) -> dict:
    """Spearman deux a deux sur l'union des top-n, et recouvrement des top-n."""
    union = sorted(set().union(*[set(s.nlargest(top_n).index) for s in importances]))
    rhos, overlaps = [], []
    for a, b in itertools.combinations(importances, 2):
        rhos.append(spearmanr(a.reindex(union).fillna(0), b.reindex(union).fillna(0)).statistic)
        overlaps.append(len(set(a.nlargest(top_n).index) & set(b.nlargest(top_n).index)) / top_n)
    return {
        "spearman_mean": float(np.mean(rhos)), "spearman_min": float(np.min(rhos)),
        "top_overlap_mean": float(np.mean(overlaps)), "top_overlap_min": float(np.min(overlaps)),
        "n_features_in_union": len(union),
        "top10_per_run": [list(s.nlargest(top_n).index) for s in importances],
    }


def expected_calibration_error(y_true, y_proba, n_bins=10) -> float:
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    bins = np.minimum((y_proba * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - y_proba[m].mean())
    return float(ece)


def group_metrics(y_true, y_proba, groups, threshold=0.5) -> pd.DataFrame:
    y_true, y_proba, groups = map(np.asarray, (y_true, y_proba, groups))
    y_pred = (y_proba >= threshold).astype(int)
    rows = []
    for g in sorted(pd.unique(groups), key=str):
        m = groups == g
        both = y_true[m].min() != y_true[m].max()
        rows.append({
            "group": str(g), "n": int(m.sum()), "n_churners": int(y_true[m].sum()),
            "churn_rate": float(y_true[m].mean()),
            "auc": float(roc_auc_score(y_true[m], y_proba[m])) if both else np.nan,
            "recall": float(recall_score(y_true[m], y_pred[m], zero_division=0)) if y_true[m].any() else np.nan,
            "precision": float(precision_score(y_true[m], y_pred[m], zero_division=0)),
        })
    return pd.DataFrame(rows)


def _xgb_with_seed(xgb_pipeline, y, seed):
    p = clone(xgb_pipeline)
    p.set_params(model__random_state=seed, model__n_jobs=1,
                 model__scale_pos_weight=compute_scale_pos_weight(y))
    return p


def write_group_performance(mdir: Path, out: Path, X_test, y_test, p_served) -> None:
    """Performance du modele servi par type de contrat et par segment (etape 3)."""
    seg_path = mdir / "segments.csv"
    by_contract = group_metrics(y_test, p_served, X_test["Contract"])
    by_contract.to_csv(out / "performance_by_contract.csv", index=False)
    print("\nPar contrat :\n" + by_contract.round(4).to_string(index=False))
    if seg_path.exists():
        seg = pd.read_csv(seg_path, index_col="row")["segment"]
        by_seg = group_metrics(y_test, p_served, seg.loc[X_test.index])
        by_seg.to_csv(out / "performance_by_segment.csv", index=False)
        print("\nPar segment (k-prototypes) :\n" + by_seg.round(4).to_string(index=False))
    else:
        print(f"{seg_path} absent : lancer `python -m src.segmentation` pour la performance par segment")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/telco_churn.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--out-dir", default="reports/robustness")
    parser.add_argument("--fig-dir", default="reports/figures")
    parser.add_argument("--segments-only", action="store_true",
                        help="ne recalcule que la performance par contrat et par segment")
    args = parser.parse_args()
    mdir, out, figs = Path(args.model_dir), Path(args.out_dir), Path(args.fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    split = joblib.load(mdir / "test_split.pkl")
    X_train, y_train, X_test, y_test = (split[k] for k in ("X_train", "y_train", "X_test", "y_test"))
    served = joblib.load(mdir / "best_model.pkl")
    if args.segments_only:
        write_group_performance(mdir, out, X_test, y_test, served.predict_proba(X_test)[:, 1])
        return
    xgb = joblib.load(mdir / "xgboost_model.pkl")
    served_name = type(served.named_steps["model"]).__name__
    origin = feature_origin(xgb.named_steps["preprocessor"])
    summary: dict = {}

    # 1. Bootstrap
    p_served = served.predict_proba(X_test)[:, 1]
    p_xgb = xgb.predict_proba(X_test)[:, 1]
    summary["bootstrap"] = {served_name: bootstrap_ci(y_test, p_served),
                            "XGBClassifier": bootstrap_ci(y_test, p_xgb)}
    for name, r in summary["bootstrap"].items():
        print(f"{name:22s} AUC {r['auc']['point']:.4f} IC95 [{r['auc']['ci95'][0]:.4f}, "
              f"{r['auc']['ci95'][1]:.4f}] | rappel {r['recall']['point']:.4f} IC95 "
              f"[{r['recall']['ci95'][0]:.4f}, {r['recall']['ci95'][1]:.4f}]")

    # 2. Stabilite SHAP : folds
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    fold_imps = []
    for tr, _ in cv.split(X_train, y_train):
        p = _xgb_with_seed(xgb, y_train.iloc[tr], RANDOM_STATE).fit(X_train.iloc[tr], y_train.iloc[tr])
        fold_imps.append(mean_abs_importance(tree_shap(p, X_test)[0], origin))
    # ... et seeds (nouveau split + nouvelle seed du modele)
    X_all, y_all = split_X_y(clean_data(load_data(args.data)))
    seed_imps = []
    for seed in SEEDS:
        Xtr, Xte, ytr, _ = train_test_split(X_all, y_all, test_size=TEST_SIZE, stratify=y_all,
                                            random_state=seed)
        p = _xgb_with_seed(xgb, ytr, seed).fit(Xtr, ytr)
        seed_imps.append(mean_abs_importance(tree_shap(p, Xte)[0], origin))
    summary["shap_stability"] = {"folds": rank_stability(fold_imps), "seeds": rank_stability(seed_imps)}
    for k, r in summary["shap_stability"].items():
        print(f"Stabilite top {TOP_N} SHAP ({k}) : Spearman moyen {r['spearman_mean']:.4f} "
              f"(min {r['spearman_min']:.4f}), recouvrement moyen {r['top_overlap_mean']:.2f} "
              f"(min {r['top_overlap_min']:.2f})")

    # 3. Calibration
    inner = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    oof = {"brut": cross_val_predict(clone(served), X_train, y_train, cv=inner,
                                     method="predict_proba", n_jobs=1)[:, 1]}
    calibrated = {}
    for method in ("isotonic", "sigmoid"):
        cal = CalibratedClassifierCV(clone(served), method=method, cv=5, ensemble=False, n_jobs=1)
        oof[method] = cross_val_predict(cal, X_train, y_train, cv=inner, method="predict_proba", n_jobs=1)[:, 1]
        calibrated[method] = clone(cal).fit(X_train, y_train)
    brier_train = {k: float(brier_score_loss(y_train, v)) for k, v in oof.items()}
    best_method = min(("isotonic", "sigmoid"), key=brier_train.get)
    use_cal = brier_train[best_method] < brier_train["brut"]
    test_proba = {"brut": p_served} | {m: c.predict_proba(X_test)[:, 1] for m, c in calibrated.items()}
    summary["calibration"] = {
        "model": served_name,
        "brier_train_oof": brier_train,
        "brier_test": {k: float(brier_score_loss(y_test, v)) for k, v in test_proba.items()},
        "ece_test": {k: expected_calibration_error(y_test, v) for k, v in test_proba.items()},
        "auc_test": {k: float(roc_auc_score(y_test, v)) for k, v in test_proba.items()},
        "chosen_method": best_method if use_cal else None,
        "recalibrated": bool(use_cal),
        "base_rate_test": float(np.mean(y_test)),
        "mean_proba_test": {k: float(np.mean(v)) for k, v in test_proba.items()},
    }
    if use_cal:
        joblib.dump(calibrated[best_method], mdir / "calibrated_model.pkl")
    print("Brier test : " + ", ".join(f"{k}={v:.4f}" for k, v in summary["calibration"]["brier_test"].items())
          + f" -> recalibration retenue : {best_method if use_cal else 'aucune'}")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="parfaite")
    for k, v in test_proba.items():
        frac, mean = calibration_curve(y_test, v, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "o-", label=f"{k} (Brier {brier_score_loss(y_test, v):.3f})")
    ax.set(xlabel="Probabilite predite", ylabel="Frequence observee de churn",
           title=f"Calibration -- {served_name}, test set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figs / "calibration_curve.png", dpi=120)
    plt.close(fig)

    # 4. Performance par segment et par contrat (modele servi, seuil 0,5)
    write_group_performance(mdir, out, X_test, y_test, p_served)

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
