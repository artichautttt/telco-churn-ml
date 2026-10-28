"""
business.py -- lecture metier (etape 6).

Seuil de decision selon une matrice de couts parametrable. Hypotheses par
defaut (a remplacer par les vrais chiffres de l'equipe marketing) :
- cout d'une action de retention (appel, remise...) : 50 EUR par client contacte ;
- perte liee a un client qui part : 500 EUR (~ 8 mois de la facture mensuelle
  moyenne du jeu, 64,8 EUR) ;
- taux de succes d'une action sur un vrai churner : 50 %.

Cout par client selon la decision :
  vrai negatif  (non contacte, reste)  : 0
  faux positif  (contacte, serait reste) : action
  faux negatif  (non contacte, part)   : perte
  vrai positif  (contacte, allait partir): action + (1 - succes) x perte

Le seuil est choisi sur les probabilites hors-fold du train set (CV 5-fold),
puis applique tel quel au test set. Si un modele recalibre existe
(models/calibrated_model.pkl, etape 5), ses probabilites sont utilisees ;
pour des probabilites bien calibrees, le seuil theorique est
action / (succes x perte).

Courbes de gains et de lift sur le test set : part des churners captes dans
les x % de clients au score le plus eleve.

Usage : python -m src.business [--action-cost 50 --churn-loss 500 --success-rate 0.5]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.train import RANDOM_STATE

THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)
SERVING_CONFIG = "serving_config.json"


@dataclass(frozen=True)
class Costs:
    action_cost: float = 50.0
    churn_loss: float = 500.0
    success_rate: float = 0.5

    def theoretical_threshold(self) -> float:
        return self.action_cost / (self.success_rate * self.churn_loss)


def total_cost(y_true, y_proba, threshold: float, costs: Costs) -> float:
    y_true = np.asarray(y_true)
    contact = np.asarray(y_proba) >= threshold
    tp = np.sum(contact & (y_true == 1))
    fp = np.sum(contact & (y_true == 0))
    fn = np.sum(~contact & (y_true == 1))
    return float(
        fp * costs.action_cost
        + fn * costs.churn_loss
        + tp * (costs.action_cost + (1 - costs.success_rate) * costs.churn_loss)
    )


def optimal_threshold(y_true, y_proba, costs: Costs, thresholds=THRESHOLDS) -> tuple[float, np.ndarray]:
    curve = np.array([total_cost(y_true, y_proba, t, costs) for t in thresholds])
    return float(thresholds[int(np.argmin(curve))]), curve


def gains_curve(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    """(part des clients contactes, part des churners captes), tri par score decroissant."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(scores), kind="stable")
    captured = np.cumsum(y_true[order]) / y_true.sum()
    frac = np.arange(1, len(y_true) + 1) / len(y_true)
    return np.concatenate([[0], frac]), np.concatenate([[0], captured])


def capture_at(y_true, scores, top_frac: float) -> float:
    """Part des churners dans les top_frac clients au score le plus eleve."""
    y_true = np.asarray(y_true)
    k = int(np.ceil(top_frac * len(y_true)))
    order = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(y_true[order].sum() / y_true.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--out-dir", default="reports/business")
    parser.add_argument("--fig-dir", default="reports/figures")
    parser.add_argument("--action-cost", type=float, default=Costs.action_cost)
    parser.add_argument("--churn-loss", type=float, default=Costs.churn_loss)
    parser.add_argument("--success-rate", type=float, default=Costs.success_rate)
    args = parser.parse_args()
    mdir, out, figs = Path(args.model_dir), Path(args.out_dir), Path(args.fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    costs = Costs(args.action_cost, args.churn_loss, args.success_rate)

    split = joblib.load(mdir / "test_split.pkl")
    X_train, y_train, X_test, y_test = (split[k] for k in ("X_train", "y_train", "X_test", "y_test"))
    cal_path = mdir / "calibrated_model.pkl"
    model_file = cal_path if cal_path.exists() else mdir / "best_model.pkl"
    model = joblib.load(model_file)
    served = joblib.load(mdir / "best_model.pkl")

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    t_opt, curve_train = optimal_threshold(y_train, oof, costs)
    p_test = model.predict_proba(X_test)[:, 1]

    def decision_stats(t):
        pred = (p_test >= t).astype(int)
        return {
            "threshold": float(t),
            "total_cost_test": total_cost(y_test, p_test, t, costs),
            "contacted_share": float(pred.mean()),
            "recall": float(recall_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
        }

    n = len(y_test)
    baseline_nobody = total_cost(y_test, p_test, 1.01, costs)
    baseline_everybody = total_cost(y_test, p_test, -0.01, costs)
    s_served = served.predict_proba(X_test)[:, 1]
    summary = {
        "costs_hypotheses": asdict(costs),
        "probabilities_from": model_file.name,
        "theoretical_threshold_if_calibrated": costs.theoretical_threshold(),
        "optimal_threshold_train_oof": t_opt,
        "at_optimal_threshold": decision_stats(t_opt),
        "at_default_threshold_0_5": decision_stats(0.5),
        "cost_contact_nobody_test": baseline_nobody,
        "cost_contact_everybody_test": baseline_everybody,
        "n_test": n,
        "n_churners_test": int(np.sum(y_test)),
        "gains_scores_from": "best_model.pkl",
        "capture_top10": capture_at(y_test, s_served, 0.10),
        "capture_top20": capture_at(y_test, s_served, 0.20),
        "lift_top10": capture_at(y_test, s_served, 0.10) / 0.10,
        "lift_top20": capture_at(y_test, s_served, 0.20) / 0.20,
        # La recalibration doit preserver le classement : meme AUC.
        "auc_test_decision_model": float(roc_auc_score(y_test, p_test)),
        "auc_test_uncalibrated": float(roc_auc_score(y_test, s_served)),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Performance de la decision servie (modele + seuil) par contrat et par segment.
    from src.robustness import group_metrics

    tables = [group_metrics(y_test, p_test, X_test["Contract"], threshold=t_opt).assign(by="contract")]
    seg_path = mdir / "segments.csv"
    if seg_path.exists():
        seg = pd.read_csv(seg_path, index_col="row")["segment"].loc[X_test.index]
        tables.append(group_metrics(y_test, p_test, seg, threshold=t_opt).assign(by="segment"))
    by_group = pd.concat(tables, ignore_index=True)
    by_group.to_csv(out / "performance_by_group_at_threshold.csv", index=False)
    print(f"\nDecision servie (seuil {t_opt:.2f}) par groupe :\n" + by_group.round(4).to_string(index=False))

    # Configuration lue par l'API : modele servi et seuil de decision.
    serving = {
        "model_path": model_file.as_posix(),
        "threshold": t_opt,
        "threshold_source": "matrice de couts, probabilites hors-fold du train (src.business)",
        "costs_hypotheses": asdict(costs),
    }
    with open(mdir / SERVING_CONFIG, "w") as f:
        json.dump(serving, f, indent=2)
    print(f"Configuration de service ecrite dans {mdir / SERVING_CONFIG} : "
          f"{model_file.name}, seuil {t_opt:.2f}")
    print(f"AUC test : modele de decision {summary['auc_test_decision_model']:.6f} | "
          f"non recalibre {summary['auc_test_uncalibrated']:.6f}")

    opt = summary["at_optimal_threshold"]
    d05 = summary["at_default_threshold_0_5"]
    print(f"Hypotheses : {asdict(costs)} ; probabilites : {model_file.name}")
    print(f"Seuil optimal (train hors-fold) = {t_opt:.2f} (theorique si calibre : "
          f"{costs.theoretical_threshold():.2f})")
    for label, s in (("seuil optimal", opt), ("seuil 0,5", d05)):
        print(f"  {label:13s} cout test = {s['total_cost_test']:.0f} EUR | contactes "
              f"{s['contacted_share']:.1%} | rappel {s['recall']:.3f} | precision {s['precision']:.3f}")
    print(f"  personne contacte : {baseline_nobody:.0f} EUR ; tout le monde : {baseline_everybody:.0f} EUR")
    print(f"Gains : top 10 % -> {summary['capture_top10']:.1%} des churners (lift "
          f"{summary['lift_top10']:.2f}) ; top 20 % -> {summary['capture_top20']:.1%} "
          f"(lift {summary['lift_top20']:.2f})")

    frac, captured = gains_curve(y_test, s_served)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frac, captured, label="modele")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="aleatoire")
    for q in (0.1, 0.2):
        axes[0].axvline(q, color="grey", lw=0.5)
    axes[0].set(xlabel="Part des clients contactes (tri par score)", ylabel="Part des churners captes",
                title="Courbe de gains -- test set")
    axes[0].legend()
    lift = np.divide(captured[1:], frac[1:])
    axes[1].plot(frac[1:], lift)
    axes[1].axhline(1, color="k", ls="--", lw=1)
    axes[1].set(xlabel="Part des clients contactes", ylabel="Lift", title="Courbe de lift -- test set")
    plt.tight_layout()
    plt.savefig(figs / "gains_lift.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(THRESHOLDS, curve_train / len(y_train), label="train (hors-fold)")
    ax.axvline(t_opt, color="C3", ls="--", label=f"seuil optimal {t_opt:.2f}")
    ax.set(xlabel="Seuil de decision", ylabel="Cout moyen par client (EUR)",
           title="Cout attendu selon le seuil")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figs / "cost_vs_threshold.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
