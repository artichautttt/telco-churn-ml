"""
drift.py -- detection de derive des donnees et des predictions.

Utilise par notebooks/03_drift_monitoring.ipynb et src/drift_report.py.
Le jeu IBM Telco est public et n'a aucune derive reelle : le notebook valide
ces fonctions sur des derives INJECTEES (fonctions inject_*), pas sur un cas
reel.

Mesures par variable :
- PSI (Population Stability Index) : sum((c - r) * ln(c / r)) sur des classes
  fixees sur la reference (deciles pour une variable numerique, modalites pour
  une categorielle). Les proportions nulles sont remplacees par `eps`, pour
  que le PSI reste fini (classe vide, modalite inconnue).
- test d'egalite des distributions : Kolmogorov-Smirnov (numerique), chi-deux
  d'homogeneite (categorielle) ; valeurs p corrigees par Benjamini-Hochberg sur
  l'ensemble des variables.
Niveau d'alerte : convention PSI 0,1 / 0,25 combinee a la valeur p corrigee
(alert_level). Ce sont des conventions d'usage, pas des regles.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.preprocessing import NUMERIC_FEATURES

EPS = 1e-4
LEVELS = ("aucun", "surveillance", "alerte")

# SeniorCitizen est un 0/1 : traite comme categoriel pour la derive (deciles
# et KS n'ont pas de sens sur une variable binaire). Le modele n'est pas modifie.
DRIFT_NUMERIC = [c for c in NUMERIC_FEATURES if c != "SeniorCitizen"]


def drift_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(numeriques, categorielles) pour la derive : SeniorCitizen en categoriel."""
    numeric = [c for c in DRIFT_NUMERIC if c in df.columns]
    categorical = [c for c in df.columns if c not in numeric and c != "Churn"]
    return numeric, categorical


@dataclass(frozen=True)
class Thresholds:
    """Seuils d'alerte. Valeurs par defaut : convention d'usage du PSI."""
    psi_watch: float = 0.10
    psi_alert: float = 0.25
    alpha: float = 0.05

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_THRESHOLDS = Thresholds()


# --- PSI ---------------------------------------------------------------------

def _psi_from_props(ref_p: np.ndarray, cur_p: np.ndarray, eps: float = EPS) -> float:
    ref_p = np.clip(ref_p, eps, None)
    cur_p = np.clip(cur_p, eps, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def numeric_bin_edges(ref, n_bins: int = 10) -> np.ndarray:
    """Bornes interieures : quantiles de la reference, doublons fusionnes."""
    ref = np.asarray(ref, dtype=float)
    ref = ref[~np.isnan(ref)]
    qs = np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.unique(qs)


def _bin_props(values, edges: np.ndarray) -> np.ndarray:
    """Proportions par classe ]-inf, e1], ]e1, e2], ..., ]ek, +inf[ (NaN exclus)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    idx = np.searchsorted(edges, values, side="left")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    return counts / max(len(values), 1)


def psi_numeric(ref, cur, n_bins: int = 10, eps: float = EPS) -> float:
    """PSI d'une variable numerique, classes fixees sur la reference."""
    edges = numeric_bin_edges(ref, n_bins)
    return _psi_from_props(_bin_props(ref, edges), _bin_props(cur, edges), eps)


def _category_props(ref, cur) -> tuple[np.ndarray, np.ndarray, list]:
    ref = pd.Series(ref).astype(str)
    cur = pd.Series(cur).astype(str)
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return r, c, cats


def psi_categorical(ref, cur, eps: float = EPS) -> float:
    """PSI d'une variable categorielle sur l'union des modalites.

    Une modalite absente de la reference (categorie inconnue) a une proportion
    de reference nulle, remplacee par eps : elle pese fortement dans le PSI.
    """
    r, c, _ = _category_props(ref, cur)
    return _psi_from_props(r, c, eps)


# --- Tests statistiques ------------------------------------------------------

def ks_test(ref, cur) -> tuple[float, float]:
    """Kolmogorov-Smirnov a deux echantillons (scipy) : (statistique, p)."""
    res = stats.ks_2samp(np.asarray(ref, dtype=float), np.asarray(cur, dtype=float))
    return float(res.statistic), float(res.pvalue)


def chi2_test(ref, cur) -> tuple[float, float]:
    """Chi-deux d'homogeneite reference / courant sur l'union des modalites."""
    ref = pd.Series(ref).astype(str)
    cur = pd.Series(cur).astype(str)
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    table = np.array([ref.value_counts().reindex(cats, fill_value=0).to_numpy(),
                      cur.value_counts().reindex(cats, fill_value=0).to_numpy()])
    if table.shape[1] < 2:
        return 0.0, 1.0
    res = stats.chi2_contingency(table, correction=False)
    return float(res.statistic), float(res.pvalue)


def bh_adjust(p_values) -> np.ndarray:
    """Valeurs p corrigees par Benjamini-Hochberg (statsmodels)."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return p
    return multipletests(p, method="fdr_bh")[1]


# --- Niveaux d'alerte --------------------------------------------------------

def alert_level(psi: float, p_adjusted: float, thresholds: Thresholds = DEFAULT_THRESHOLDS) -> str:
    """Convention : alerte si PSI >= psi_alert ET p corrigee < alpha ;
    surveillance si PSI >= psi_watch OU p corrigee < alpha ; sinon aucun.

    Le PSI mesure la taille de l'ecart ; la valeur p dit s'il depasse le
    hasard. Avec de gros echantillons, un ecart minuscule peut etre
    significatif : il ne declenche alors qu'une surveillance.
    """
    significant = p_adjusted < thresholds.alpha
    if psi >= thresholds.psi_alert and significant:
        return "alerte"
    if psi >= thresholds.psi_watch or significant:
        return "surveillance"
    return "aucun"


def feature_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame, numeric: list[str] | None = None,
                  categorical: list[str] | None = None, thresholds: Thresholds = DEFAULT_THRESHOLDS,
                  n_bins: int = 10) -> pd.DataFrame:
    """Une ligne par variable : PSI, test, statistique, p, p corrigee (BH), niveau."""
    if numeric is None or categorical is None:
        numeric, categorical = drift_columns(ref_df)
    rows = []
    for col in numeric:
        stat, p = ks_test(ref_df[col], cur_df[col])
        rows.append({"variable": col, "type": "numerique", "test": "ks",
                     "psi": psi_numeric(ref_df[col], cur_df[col], n_bins), "statistic": stat, "p_value": p})
    for col in categorical:
        stat, p = chi2_test(ref_df[col], cur_df[col])
        rows.append({"variable": col, "type": "categorielle", "test": "chi2",
                     "psi": psi_categorical(ref_df[col], cur_df[col]), "statistic": stat, "p_value": p})
    out = pd.DataFrame(rows)
    out["p_adjusted"] = bh_adjust(out["p_value"])
    out["level"] = [alert_level(s, p, thresholds) for s, p in zip(out["psi"], out["p_adjusted"])]
    return out


def prediction_drift(p_ref, p_cur, threshold: float, thresholds: Thresholds = DEFAULT_THRESHOLDS,
                     n_bins: int = 10) -> dict:
    """Derive des predictions : PSI et KS sur la probabilite, part au-dessus du seuil.

    Une seule mesure (pas de correction multiple) : le niveau utilise la valeur p du KS.
    """
    p_ref, p_cur = np.asarray(p_ref, dtype=float), np.asarray(p_cur, dtype=float)
    psi = psi_numeric(p_ref, p_cur, n_bins)
    stat, p = ks_test(p_ref, p_cur)
    share_ref, share_cur = float((p_ref >= threshold).mean()), float((p_cur >= threshold).mean())
    return {"psi": psi, "ks_statistic": stat, "p_value": p, "threshold": threshold,
            "share_above_ref": share_ref, "share_above_cur": share_cur,
            "share_above_diff": share_cur - share_ref, "level": alert_level(psi, p, thresholds)}


def overall_level(levels) -> str:
    """Niveau le plus eleve d'une liste de niveaux."""
    return max(levels, key=LEVELS.index, default="aucun")


# --- Derives injectees (SIMULEES) --------------------------------------------

def inject_tenure_shift(df: pd.DataFrame, months: float) -> pd.DataFrame:
    """Anciennete baissee de `months` (bornee a 0) : afflux simule de clients recents.
    TotalCharges est ajuste au prorata de l'anciennete pour rester coherent."""
    out = df.copy()
    old = out["tenure"].astype(float)
    new = np.maximum(old - months, 0)
    ratio = np.where(old > 0, new / old.where(old > 0, 1), 1.0)
    out["tenure"] = new.astype(df["tenure"].dtype) if months == int(months) else new
    out["TotalCharges"] = out["TotalCharges"] * ratio
    return out


def inject_monthly_charges_increase(df: pd.DataFrame, pct: float) -> pd.DataFrame:
    """Hausse tarifaire simulee de `pct` (0,05 = +5 %) sur MonthlyCharges.
    TotalCharges (historique deja facture) n'est pas modifie."""
    out = df.copy()
    out["MonthlyCharges"] = out["MonthlyCharges"] * (1 + pct)
    return out


def inject_contract_mix(df: pd.DataFrame, month_to_month_share: float, seed) -> pd.DataFrame:
    """Part de contrats mensuels portee a `month_to_month_share` en reaffectant au
    hasard des clients a contrat 1 ou 2 ans. Seule la colonne Contract change :
    les autres variables (anciennete, prix) ne suivent pas, ce qui est une limite."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    mtm = out["Contract"] == "Month-to-month"
    needed = round(month_to_month_share * len(out)) - int(mtm.sum())
    if needed > 0:
        candidates = out.index[~mtm].to_numpy()
        chosen = rng.choice(candidates, size=min(needed, len(candidates)), replace=False)
        out.loc[chosen, "Contract"] = "Month-to-month"
    return out


def inject_concept_drift(X: pd.DataFrame, y, flip_share: float, tenure_cut: float, seed) -> np.ndarray:
    """Derive conceptuelle simulee : X inchange, relation anciennete-churn modifiee.
    Parmi les clients d'anciennete > tenure_cut, une part `flip_share` des
    non-churners devient churner. Retourne les nouveaux labels."""
    y = np.asarray(y).astype(int).copy()
    rng = np.random.default_rng(seed)
    eligible = (X["tenure"].to_numpy() > tenure_cut) & (y == 0)
    y[eligible & (rng.random(len(y)) < flip_share)] = 1
    return y


# --- Performance a etiquettes retardees --------------------------------------

def performance_at_threshold(y, proba, threshold: float) -> dict:
    """Rappel et precision de la decision servie (contact si proba >= seuil)."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(proba) >= threshold
    tp = int((pred & (y == 1)).sum())
    return {"recall": tp / max(int(y.sum()), 1), "precision": tp / max(int(pred.sum()), 1),
            "contacted_share": float(pred.mean()), "churn_rate": float(y.mean())}


# --- Tirages repetes ---------------------------------------------------------

def random_split(n_total: int, n_current: int, seed) -> tuple[np.ndarray, np.ndarray]:
    """Redecoupage aleatoire des indices en (reference, courant), sans remise."""
    perm = np.random.default_rng(seed).permutation(n_total)
    return np.sort(perm[n_current:]), np.sort(perm[:n_current])


def rate_with_se(successes: int, n: int) -> dict:
    """Taux et erreur-type binomiale sqrt(p(1 - p) / n)."""
    rate = successes / n if n else float("nan")
    return {"rate": rate, "se": float(np.sqrt(rate * (1 - rate) / n)) if n else float("nan"), "n": n}


def evaluate_draw(ref_df: pd.DataFrame, cur_df: pd.DataFrame, p_ref, p_cur, threshold: float,
                  thresholds: Thresholds = DEFAULT_THRESHOLDS) -> dict:
    """Une comparaison reference / courant : niveaux par variable et des predictions."""
    table = feature_drift(ref_df, cur_df, thresholds=thresholds)
    pred = prediction_drift(p_ref, p_cur, threshold, thresholds)
    return {"features": table, "prediction": pred,
            "features_level": overall_level(table["level"]),
            "overall_level": overall_level([*table["level"], pred["level"]])}


def choose_psi_threshold(null_max_psi, target_rate: float, convention: float) -> float:
    """Seuil PSI fige a partir des seules comparaisons SANS derive.

    null_max_psi : PSI maximal (toutes variables) de chaque redecoupage
    reference contre reference. On garde la convention si elle donne au plus
    `target_rate` de fausses alertes ; sinon on la remonte au quantile
    1 - target_rate de cette distribution. Les scenarios injectes ne servent
    jamais a ce choix.
    """
    q = float(np.quantile(np.asarray(null_max_psi, dtype=float), 1 - target_rate))
    return max(convention, q)


# --- Version vectorisee pour les tirages repetes -----------------------------

class DriftSimulator:
    """Meme calcul que feature_drift + prediction_drift, pour des milliers de tirages.

    Les modalites sont codees une fois en entiers (comptages par np.bincount).
    La reference d'un tirage est une partie des lignes de `ref_pool` ; le
    courant, une partie des lignes d'un jeu encode par `encode` (le meme pool,
    eventuellement modifie par une derive injectee). Equivalence avec
    feature_drift / prediction_drift verifiee par tests/test_drift.py.
    """

    def __init__(self, ref_pool: pd.DataFrame, p_ref_pool, threshold: float, n_bins: int = 10):
        self.numeric, self.categorical = drift_columns(ref_pool)
        self.variables = [*self.numeric, *self.categorical]
        self.categories = {c: sorted(ref_pool[c].astype(str).unique()) for c in self.categorical}
        self.threshold, self.n_bins = threshold, n_bins
        self.ref = self.encode(ref_pool)
        self.p_ref = np.asarray(p_ref_pool, dtype=float)

    def encode(self, df: pd.DataFrame) -> dict:
        """Numeriques en float, categorielles en codes (modalite inconnue -> code a part)."""
        out = {c: df[c].to_numpy(dtype=float) for c in self.numeric}
        for c in self.categorical:
            values = df[c].astype(str).to_numpy()
            known = {v: i for i, v in enumerate(self.categories[c])}
            extra = sorted(set(values) - known.keys())
            known.update({v: len(self.categories[c]) + i for i, v in enumerate(extra)})
            out[c] = np.array([known[v] for v in values], dtype=int)
        return out

    def _categorical(self, r: np.ndarray, c: np.ndarray) -> tuple[float, float]:
        k = max(r.max(initial=0), c.max(initial=0)) + 1
        rc, cc = np.bincount(r, minlength=k), np.bincount(c, minlength=k)
        psi = _psi_from_props(rc / len(r), cc / len(c))
        keep = (rc + cc) > 0
        if keep.sum() < 2:
            return psi, 1.0
        # chi-deux de Pearson sur la table 2 x k (meme resultat que chi2_contingency, sans sa surcharge)
        table = np.array([rc[keep], cc[keep]], dtype=float)
        expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
        chi2 = float(((table - expected) ** 2 / expected).sum())
        return psi, float(stats.chi2.sf(chi2, table.shape[1] - 1))

    def evaluate(self, ref_idx, cur: dict, p_cur_pool, cur_idx,
                 thresholds: Thresholds = DEFAULT_THRESHOLDS) -> dict:
        psi, p = [], []
        for col in self.numeric:
            r, c = self.ref[col][ref_idx], cur[col][cur_idx]
            psi.append(psi_numeric(r, c, self.n_bins))
            p.append(ks_test(r, c)[1])
        for col in self.categorical:
            value, pv = self._categorical(self.ref[col][ref_idx], cur[col][cur_idx])
            psi.append(value)
            p.append(pv)
        p_adj = bh_adjust(p)
        levels = [alert_level(s, q, thresholds) for s, q in zip(psi, p_adj)]
        pred = prediction_drift(self.p_ref[ref_idx], np.asarray(p_cur_pool)[cur_idx], self.threshold,
                                thresholds, self.n_bins)
        return {"psi": np.array(psi), "p_value": np.array(p), "p_adjusted": p_adj, "levels": levels,
                "prediction": pred, "features_level": overall_level(levels),
                "overall_level": overall_level([*levels, pred["level"]])}
