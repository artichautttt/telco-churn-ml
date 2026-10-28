"""
segmentation.py -- typologies de clients (etape 3).

- k-prototypes (donnees mixtes : numeriques standardisees + categorielles
  brutes), compare a un K-means sur variables one-hot + standardisees ;
- choix de k sur k = 2..8 : courbe de cout (coude) et silhouette. Pour
  k-prototypes, la silhouette est calculee sur une distance de Gower
  (echantillon) ; pour K-means, sur la distance euclidienne de l'espace
  encode. Regle : k = silhouette maximale parmi 3..5 (le besoin metier est de
  3 a 5 profils lisibles) ; la courbe de cout est rapportee pour controle ;
- profils : taille, taux de churn, variables qui distinguent chaque segment ;
- classement des facteurs SHAP (XGBoost) par segment, compare au classement
  global (correlation de Spearman).

La segmentation n'utilise pas la cible Churn : le taux de churn est decrit
a posteriori.

Usage : python -m src.segmentation
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
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import TARGET, clean_data, load_data, split_X_y
from src.shap_utils import aggregate_by_feature, feature_origin, tree_shap

RANDOM_STATE = 42
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
K_RANGE = range(2, 9)
K_CHOICES = (3, 4, 5)
SILHOUETTE_SAMPLE = 2000


def split_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Numeriques continues vs categorielles (SeniorCitizen, 0/1, est categoriel)."""
    num = [c for c in NUMERIC if c in X.columns]
    cat = [c for c in X.columns if c not in num]
    return num, cat


def gower_distance(num: np.ndarray, cat: np.ndarray) -> np.ndarray:
    """Distance de Gower : |ecart| / etendue pour les numeriques, 0/1 (different)
    pour les categorielles, moyenne sur toutes les variables."""
    n = len(num) if num.size else len(cat)
    d = np.zeros((n, n))
    ranges = num.max(axis=0) - num.min(axis=0) if num.size else []
    for j in range(num.shape[1] if num.size else 0):
        r = ranges[j] if ranges[j] > 0 else 1.0
        d += np.abs(num[:, j, None] - num[None, :, j]) / r
    for j in range(cat.shape[1] if cat.size else 0):
        d += cat[:, j, None] != cat[None, :, j]
    p = (num.shape[1] if num.size else 0) + (cat.shape[1] if cat.size else 0)
    return d / p


def encode_for_kmeans(X: pd.DataFrame) -> np.ndarray:
    num, cat = split_types(X)
    a = StandardScaler().fit_transform(X[num])
    b = OneHotEncoder(sparse_output=False).fit_transform(X[cat].astype(str))
    b = StandardScaler().fit_transform(b)
    return np.hstack([a, b])


def fit_kprototypes(X: pd.DataFrame, k: int, n_init: int = 3):
    from kmodes.kprototypes import KPrototypes

    num, cat = split_types(X)
    data = pd.concat(
        [pd.DataFrame(StandardScaler().fit_transform(X[num]), columns=num, index=X.index),
         X[cat].astype(str)],
        axis=1,
    )
    cat_idx = list(range(len(num), len(num) + len(cat)))
    model = KPrototypes(n_clusters=k, init="Cao", n_init=n_init, random_state=RANDOM_STATE)
    labels = model.fit_predict(data.to_numpy(dtype=object), categorical=cat_idx)
    return labels, float(model.cost_)


def choose_k(silhouettes: dict[int, float], choices=K_CHOICES) -> int:
    """k a silhouette maximale parmi les valeurs metier admissibles."""
    cands = {k: s for k, s in silhouettes.items() if k in choices}
    if not cands:
        raise ValueError("aucun k admissible evalue")
    return max(cands, key=cands.get)


def profile_segments(X: pd.DataFrame, y: pd.Series, labels: np.ndarray, top: int = 4) -> pd.DataFrame:
    """Taille, taux de churn et variables distinctives de chaque segment.

    Numeriques : ecart a la moyenne globale en ecarts-types. Categorielles :
    modalite la plus sur-representee (lift = part dans le segment / part globale,
    parmi les modalites representant au moins 30 % du segment)."""
    num, cat = split_types(X)
    rows = []
    for s in sorted(np.unique(labels)):
        m = labels == s
        traits, seen = [], set()
        for c in num:
            z = (X.loc[m, c].mean() - X[c].mean()) / X[c].std()
            mean_seg, mean_all = X.loc[m, c].mean(), X[c].mean()
            label = f"{c} {'+' if z > 0 else '-'}{abs(z):.1f} ET (moy. {mean_seg:.1f} vs {mean_all:.1f})"
            traits.append((abs(z), label))
        for c in cat:
            seg = X.loc[m, c].astype(str).value_counts(normalize=True)
            glob = X[c].astype(str).value_counts(normalize=True)
            lift = (seg / glob).loc[seg[seg >= 0.3].index]
            if len(lift):
                v = lift.idxmax()
                # Les 6 options internet valent "No internet service" ensemble :
                # un seul trait suffit.
                no_inet = v == "No internet service" or (c == "InternetService" and v == "No")
                key = "no_internet" if no_inet else c
                if key in seen:
                    continue
                seen.add(key)
                label = "sans internet" if key == "no_internet" else f"{c}={v}"
                traits.append((abs(np.log(lift[v])), f"{label} ({seg[v]:.0%} vs {glob[v]:.0%})"))
        traits.sort(reverse=True)
        rows.append({
            "segment": int(s),
            "n_clients": int(m.sum()),
            "share": round(float(m.mean()), 4),
            "churn_rate": round(float(y[m].mean()), 4),
            "share_of_churners": round(float(y[m].sum() / y.sum()), 4),
            "tenure_mean": round(float(X.loc[m, "tenure"].mean()), 1),
            "monthly_charges_mean": round(float(X.loc[m, "MonthlyCharges"].mean()), 1),
            "month_to_month_share": round(float((X.loc[m, "Contract"] == "Month-to-month").mean()), 4),
            "fiber_share": round(float((X.loc[m, "InternetService"] == "Fiber optic").mean()), 4),
            "distinctive": " ; ".join(t for _, t in traits[:top]),
        })
    return pd.DataFrame(rows)


def shap_rank_by_segment(shap_agg: pd.DataFrame, labels: np.ndarray, top: int = 5) -> pd.DataFrame:
    """Top facteurs SHAP par segment et Spearman avec le classement global."""
    glob = shap_agg.abs().mean()
    rows = [{"segment": "global", "spearman_vs_global": 1.0,
             "top_factors": ", ".join(glob.sort_values(ascending=False).index[:top])}]
    for s in sorted(np.unique(labels)):
        imp = shap_agg[labels == s].abs().mean()
        rho = spearmanr(imp, glob).statistic
        rows.append({
            "segment": int(s),
            "spearman_vs_global": round(float(rho), 4),
            "top_factors": ", ".join(imp.sort_values(ascending=False).index[:top]),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/telco_churn.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--out-dir", default="reports/segmentation")
    parser.add_argument("--fig-dir", default="reports/figures")
    args = parser.parse_args()
    out, figs, mdir = Path(args.out_dir), Path(args.fig_dir), Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    df = clean_data(load_data(args.data))
    X, y = split_X_y(df)
    num, cat = split_types(X)

    rng = np.random.default_rng(RANDOM_STATE)
    sample = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)
    num_scaled = StandardScaler().fit_transform(X[num])
    gower = gower_distance(num_scaled[sample], X[cat].astype(str).to_numpy()[sample])
    X_km = encode_for_kmeans(X)

    rows, kp_labels, km_labels = [], {}, {}
    for k in K_RANGE:
        lab_kp, cost = fit_kprototypes(X, k)
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_km)
        kp_labels[k], km_labels[k] = lab_kp, km.labels_
        sil_kp = silhouette_score(gower, lab_kp[sample], metric="precomputed")
        sil_km = silhouette_score(X_km[sample], km.labels_[sample])
        rows.append({"k": k, "kproto_cost": cost, "kproto_silhouette_gower": sil_kp,
                     "kmeans_inertia": float(km.inertia_), "kmeans_silhouette": sil_km})
        print(f"k={k}  kproto cost={cost:.0f} sil={sil_kp:.4f} | "
              f"kmeans inertia={km.inertia_:.0f} sil={sil_km:.4f}")
    ksel = pd.DataFrame(rows)
    ksel.to_csv(out / "k_selection.csv", index=False)

    k_kp = choose_k(dict(zip(ksel.k, ksel.kproto_silhouette_gower)))
    k_km = choose_k(dict(zip(ksel.k, ksel.kmeans_silhouette)))
    labels, labels_km = kp_labels[k_kp], km_labels[k_km]
    ari = adjusted_rand_score(labels, labels_km)
    print(f"\nk retenu : k-prototypes k={k_kp}, K-means k={k_km} ; ARI entre les deux = {ari:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ksel.k, ksel.kproto_cost / ksel.kproto_cost.max(), "o-", label="k-prototypes (cout)")
    axes[0].plot(ksel.k, ksel.kmeans_inertia / ksel.kmeans_inertia.max(), "s-", label="K-means (inertie)")
    axes[0].set(title="Cout normalise (coude)", xlabel="k")
    axes[1].plot(ksel.k, ksel.kproto_silhouette_gower, "o-", label="k-prototypes (Gower)")
    axes[1].plot(ksel.k, ksel.kmeans_silhouette, "s-", label="K-means (euclidienne)")
    axes[1].set(title="Silhouette", xlabel="k")
    for ax in axes:
        ax.legend()
        ax.axvspan(min(K_CHOICES) - 0.5, max(K_CHOICES) + 0.5, color="grey", alpha=0.1)
    plt.tight_layout()
    plt.savefig(figs / "segmentation_k_selection.png", dpi=120)
    plt.close(fig)

    profiles = profile_segments(X, y, labels)
    profiles_km = profile_segments(X, y, labels_km)
    profiles.to_csv(out / "profiles_kprototypes.csv", index=False)
    profiles_km.to_csv(out / "profiles_kmeans.csv", index=False)
    print("\nProfils k-prototypes :\n" + profiles.to_string(index=False))
    print("\nProfils K-means :\n" + profiles_km.to_string(index=False))

    xgb = joblib.load(mdir / "xgboost_model.pkl")
    values, _, _, _ = tree_shap(xgb, X)
    shap_agg = aggregate_by_feature(values, feature_origin(xgb.named_steps["preprocessor"]))
    shap_seg = shap_rank_by_segment(shap_agg, labels)
    shap_seg.to_csv(out / "shap_by_segment.csv", index=False)
    print("\nFacteurs SHAP (XGBoost) par segment k-prototypes :\n" + shap_seg.to_string(index=False))

    pd.DataFrame({"segment": labels, "segment_kmeans": labels_km, TARGET: y.values},
                 index=X.index).to_csv(mdir / "segments.csv", index_label="row")
    with open(out / "summary.json", "w") as f:
        json.dump({"k_kprototypes": int(k_kp), "k_kmeans": int(k_km),
                   "adjusted_rand_index": float(ari), "rule": "max silhouette for k in 3..5",
                   "silhouette_sample": SILHOUETTE_SAMPLE, "random_state": RANDOM_STATE}, f, indent=2)


if __name__ == "__main__":
    main()
