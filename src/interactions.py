"""
interactions.py -- interactions entre variables (etape 4).

- valeurs d'interaction SHAP (TreeExplainer) du pipeline XGBoost sur le test
  set, agregees par variable d'origine (somme des blocs one-hot) ;
- classement des paires par force d'interaction moyenne |phi_ij + phi_ji| ;
- dependence plots croises : tenure x Contract, MonthlyCharges x
  InternetService, et les 3 paires les plus fortes ;
- comparaison avec la regression logistique : coefficients, pente de l'effet
  de tenure par type de contrat (constante en logistique, variable en
  XGBoost), et gain d'AUC CV quand on ajoute explicitement ces deux termes
  d'interaction a la logistique.

Usage : python -m src.interactions
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import clean_data, get_feature_columns, load_data
from src.shap_utils import feature_origin, transform
from src.train import RANDOM_STATE

REQUESTED_PAIRS = [("tenure", "Contract"), ("MonthlyCharges", "InternetService")]


def aggregate_interactions(inter: np.ndarray, origin: list[str]) -> tuple[np.ndarray, list[str]]:
    """(n, p, p) colonnes transformees -> (n, q, q) variables d'origine."""
    feats = list(dict.fromkeys(origin))
    idx = np.array([feats.index(o) for o in origin])
    m = np.zeros((len(origin), len(feats)))
    m[np.arange(len(origin)), idx] = 1.0
    return m.T @ inter @ m, feats


def rank_pairs(agg: np.ndarray, feats: list[str]) -> pd.DataFrame:
    """Force d'une paire (i, j), i < j : moyenne sur les clients de |phi_ij + phi_ji|."""
    strength = np.abs(agg + agg.transpose(0, 2, 1)).mean(axis=0)
    rows = [
        {"feature_a": feats[i], "feature_b": feats[j], "strength": float(strength[i, j])}
        for i in range(len(feats)) for j in range(i + 1, len(feats))
    ]
    return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)


def slope_by_group(x: np.ndarray, contrib: np.ndarray, groups: np.ndarray) -> dict:
    """Pente (moindres carres) de la contribution en fonction de x, par groupe."""
    out = {}
    for g in pd.unique(groups):
        m = groups == g
        if m.sum() >= 3 and np.ptp(x[m]) > 0:
            out[str(g)] = float(np.polyfit(x[m], contrib[m], 1)[0])
    return out


class InteractionTerms(BaseEstimator, TransformerMixin):
    """Ajoute des colonnes num x indicatrice(categorie) aux features brutes,
    pour donner a la regression logistique des interactions explicites."""

    def __init__(self, pairs=tuple(REQUESTED_PAIRS)):
        self.pairs = pairs

    def fit(self, X, y=None):
        self.levels_ = {cat: sorted(X[cat].astype(str).unique()) for _, cat in self.pairs}
        return self

    def transform(self, X):
        X = X.copy()
        for num, cat in self.pairs:
            for lvl in self.levels_[cat][1:]:
                X[f"{num}_x_{cat}={lvl}"] = X[num] * (X[cat].astype(str) == lvl)
        return X


def interaction_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Comme build_preprocessor, avec les colonnes d'interaction standardisees
    comme variables numeriques."""
    numeric, categorical = get_feature_columns(df)
    extra = [c for c in InteractionTerms().fit(df).transform(df).columns if "_x_" in c]
    return ColumnTransformer([
        ("num", StandardScaler(), numeric + extra),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical),
    ])


def _jitter_codes(values: pd.Series, rng) -> tuple[np.ndarray, list[str]]:
    if pd.api.types.is_numeric_dtype(values):
        return values.to_numpy(dtype=float), []
    cats = sorted(values.astype(str).unique())
    codes = values.astype(str).map({c: i for i, c in enumerate(cats)}).to_numpy(dtype=float)
    return codes + rng.uniform(-0.25, 0.25, len(codes)), cats


def dependence_plot(X_raw, agg, feats, a, b, path: Path):
    """Gauche : contribution SHAP totale de a (effet propre + interactions)
    selon la valeur de a, colore par b. Droite : interaction a x b seule."""
    rng = np.random.default_rng(RANDOM_STATE)
    i, j = feats.index(a), feats.index(b)
    total = agg[:, i, :].sum(axis=1)
    pair = agg[:, i, j] + agg[:, j, i]
    x, xcats = _jitter_codes(X_raw[a], rng)
    colors = X_raw[b]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, yv, title in ((axes[0], total, f"SHAP de {a} (total)"),
                          (axes[1], pair, f"Interaction SHAP {a} x {b}")):
        if pd.api.types.is_numeric_dtype(colors):
            sc = ax.scatter(x, yv, c=colors, cmap="viridis", s=8, alpha=0.6)
            fig.colorbar(sc, ax=ax, label=b)
        else:
            for k, lvl in enumerate(sorted(colors.astype(str).unique())):
                m = colors.astype(str).to_numpy() == lvl
                ax.scatter(x[m], yv[m], s=8, alpha=0.6, label=lvl, color=f"C{k}")
            ax.legend(title=b, fontsize=8, markerscale=2)
        if xcats:
            ax.set_xticks(range(len(xcats)), xcats, rotation=20, fontsize=8)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set(xlabel=a, ylabel="log-odds", title=title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def main():
    import shap

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/telco_churn.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--out-dir", default="reports/interactions")
    parser.add_argument("--fig-dir", default="reports/figures/interactions")
    args = parser.parse_args()
    mdir, out, figs = Path(args.model_dir), Path(args.out_dir), Path(args.fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    split = joblib.load(mdir / "test_split.pkl")
    X_test, X_train, y_train = split["X_test"], split["X_train"], split["y_train"]
    xgb = joblib.load(mdir / "xgboost_model.pkl")
    lr = joblib.load(mdir / "best_model.pkl")

    Xt, _ = transform(xgb, X_test)
    inter = np.asarray(shap.TreeExplainer(xgb.named_steps["model"]).shap_interaction_values(Xt))
    agg, feats = aggregate_interactions(inter, feature_origin(xgb.named_steps["preprocessor"]))
    ranking = rank_pairs(agg, feats)
    ranking.to_csv(out / "interaction_ranking.csv", index=False)
    print("Top 10 interactions (XGBoost, test set) :\n" + ranking.head(10).to_string())

    top3 = list(ranking.head(3)[["feature_a", "feature_b"]].itertuples(index=False, name=None))
    pairs = REQUESTED_PAIRS + [p for p in top3 if p not in REQUESTED_PAIRS
                               and p[::-1] not in REQUESTED_PAIRS]
    for a, b in pairs:
        dependence_plot(X_test, agg, feats, a, b, figs / f"dependence_{a}_x_{b}.png")

    # Effet de tenure par type de contrat : XGBoost vs regression logistique.
    i_ten = feats.index("tenure")
    xgb_tenure = agg[:, i_ten, :].sum(axis=1)
    contract = X_test["Contract"].to_numpy()
    tenure = X_test["tenure"].to_numpy(dtype=float)
    lr_names = [n.split("__", 1)[-1] for n in lr.named_steps["preprocessor"].get_feature_names_out()]
    coefs = pd.Series(lr.named_steps["model"].coef_[0], index=lr_names)
    scaler = lr.named_steps["preprocessor"].named_transformers_["num"]
    num_cols = list(lr.named_steps["preprocessor"].transformers_[0][2])
    tenure_scale = scaler.scale_[num_cols.index("tenure")]
    lr_slope = float(coefs["tenure"] / tenure_scale)  # log-odds par mois, identique pour tout contrat
    xgb_slopes = slope_by_group(tenure, xgb_tenure, contract)

    # Logistique avec termes d'interaction explicites, meme CV que train.py.
    df = clean_data(load_data(args.data))
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    base = clone(lr)
    with_inter = Pipeline([
        ("interactions", InteractionTerms()),
        ("preprocessor", interaction_preprocessor(df)),
        ("model", clone(lr.named_steps["model"])),
    ])
    auc_base = cross_val_score(base, X_train, y_train, cv=cv, scoring="roc_auc")
    auc_inter = cross_val_score(with_inter, X_train, y_train, cv=cv, scoring="roc_auc")

    lr_related = coefs[[n for n in coefs.index
                        if n.startswith(("tenure", "Contract", "MonthlyCharges", "InternetService"))]]
    summary = {
        "top_interactions": ranking.head(10).to_dict(orient="records"),
        "plotted_pairs": [list(p) for p in pairs],
        "tenure_effect_slope_log_odds_per_month": {
            "logistic_regression_all_contracts": lr_slope,
            "xgboost_by_contract": xgb_slopes,
        },
        "logistic_coefficients_related": lr_related.round(4).to_dict(),
        "logistic_cv_auc": {
            "baseline_mean": float(auc_base.mean()), "baseline_std": float(auc_base.std()),
            "with_interaction_terms_mean": float(auc_inter.mean()),
            "with_interaction_terms_std": float(auc_inter.std()),
            "interaction_terms": [f"{a} x {b}" for a, b in REQUESTED_PAIRS],
        },
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPente de l'effet de tenure (log-odds / mois) -- logistique : {lr_slope:.4f} "
          f"pour tous les contrats ; XGBoost par contrat : "
          + ", ".join(f"{k}={v:.4f}" for k, v in xgb_slopes.items()))
    print(f"AUC CV logistique : {auc_base.mean():.4f} (+/-{auc_base.std():.4f}) ; "
          f"avec termes d'interaction : {auc_inter.mean():.4f} (+/-{auc_inter.std():.4f})")
    print(f"Figures dans {figs}/")


if __name__ == "__main__":
    main()
