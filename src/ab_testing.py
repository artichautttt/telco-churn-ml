"""
ab_testing.py -- fonctions d'un test A/B sur une proportion (taux de churn).

Utilise par notebooks/02_ab_test_retention.ipynb. Aucune experience reelle
n'a eu lieu : le notebook applique ces fonctions a un effet de campagne
SIMULE (simulate_treatment_effect) sur le jeu public IBM Telco.

Conventions :
- groupes : tableau booleen `treated` (True = traitement, False = controle) ;
- y : 0/1, 1 = churn ;
- difference = taux traitement - taux controle (negative = moins de churn) ;
- `seed` : un entier ou un np.random.Generator (np.random.default_rng).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_effectsize,
    proportions_ztest,
)


def target_mask(proba, threshold: float) -> np.ndarray:
    """Clients cibles par la campagne : probabilite de churn >= seuil (meme regle que l'API)."""
    return np.asarray(proba) >= threshold


# --- Dimensionnement ---------------------------------------------------------

def _check_rate(p: float, name: str) -> None:
    if not 0 < p < 1:
        raise ValueError(f"{name} doit etre dans ]0, 1[, recu {p}")


def required_sample_size(
    p_control: float, relative_reduction: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Taille par groupe (1:1) pour detecter p_control -> p_control * (1 - reduction).

    Test bilateral de deux proportions, taille d'effet h de Cohen
    (statsmodels NormalIndPower), arrondi a l'entier superieur.
    """
    _check_rate(p_control, "p_control")
    if not 0 < relative_reduction < 1:
        raise ValueError("relative_reduction doit etre dans ]0, 1[")
    h = proportion_effectsize(p_control, p_control * (1 - relative_reduction))
    n = NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power, ratio=1.0,
                                     alternative="two-sided")
    return int(np.ceil(n))


def minimum_detectable_effect(
    p_control: float, n_per_group: int, alpha: float = 0.05, power: float = 0.8
) -> dict:
    """Plus petite baisse du churn detectable avec n_per_group clients par groupe."""
    _check_rate(p_control, "p_control")
    h = NormalIndPower().solve_power(nobs1=n_per_group, alpha=alpha, power=power, ratio=1.0,
                                     alternative="two-sided")
    # h = 2 asin(sqrt(p_c)) - 2 asin(sqrt(p_t))  =>  p_t
    p_treatment = float(np.sin(np.arcsin(np.sqrt(p_control)) - h / 2) ** 2)
    return {
        "cohen_h": float(h),
        "p_control": p_control,
        "p_treatment": p_treatment,
        "absolute": p_treatment - p_control,
        "relative_reduction": 1 - p_treatment / p_control,
    }


def theoretical_power(
    p_control: float, relative_reduction: float, n_per_group: int, alpha: float = 0.05
) -> float:
    """Puissance theorique (h de Cohen, test bilateral) pour une reduction relative donnee."""
    if relative_reduction == 0:
        return alpha
    h = proportion_effectsize(p_control, p_control * (1 - relative_reduction))
    return float(NormalIndPower().power(effect_size=h, nobs1=n_per_group, alpha=alpha, ratio=1.0,
                                        alternative="two-sided"))


# --- Randomisation -----------------------------------------------------------

def assign_groups(n: int, seed, p_treatment: float = 0.5) -> np.ndarray:
    """Assignation aleatoire independante (Bernoulli) : True = traitement.

    Un tirage par client, et non un decoupage exact 50/50 : la taille des
    groupes est alors aleatoire et le controle du ratio (srm_test) a un sens.
    """
    return np.random.default_rng(seed).random(n) < p_treatment


def srm_test(n_control: int, n_treatment: int, p_treatment: float = 0.5) -> dict:
    """Controle du ratio d'echantillon (sample ratio mismatch) : chi-deux d'adequation."""
    total = n_control + n_treatment
    expected = [total * (1 - p_treatment), total * p_treatment]
    res = stats.chisquare([n_control, n_treatment], f_exp=expected)
    return {"n_control": n_control, "n_treatment": n_treatment,
            "chi2": float(res.statistic), "p_value": float(res.pvalue)}


def balance_table(df: pd.DataFrame, treated, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Equilibre des groupes : moyennes ou parts, ecart standardise (SMD) et valeur p.

    SMD = (traitement - controle) / sqrt((var_t + var_c) / 2) ; |SMD| < 0,1 est
    l'usage courant pour un bon equilibre. Valeur p : t de Welch (numerique),
    chi-deux d'independance (categoriel, une valeur par variable).
    """
    treated = np.asarray(treated)
    rows = []
    for col in numeric:
        t, c = df.loc[treated, col].astype(float), df.loc[~treated, col].astype(float)
        smd = (t.mean() - c.mean()) / np.sqrt((t.var() + c.var()) / 2)
        p = stats.ttest_ind(t, c, equal_var=False).pvalue
        rows.append({"variable": col, "level": "moyenne", "control": c.mean(), "treatment": t.mean(),
                     "smd": smd, "p_value": p})
    for col in categorical:
        table = pd.crosstab(df[col], treated)
        p = stats.chi2_contingency(table).pvalue
        for level in table.index:
            pt = (df.loc[treated, col] == level).mean()
            pc = (df.loc[~treated, col] == level).mean()
            denom = np.sqrt((pt * (1 - pt) + pc * (1 - pc)) / 2)
            rows.append({"variable": col, "level": str(level), "control": pc, "treatment": pt,
                         "smd": (pt - pc) / denom if denom > 0 else 0.0, "p_value": p})
    return pd.DataFrame(rows)


# --- Effet simule ------------------------------------------------------------

def simulate_treatment_effect(y, treated, relative_reduction: float, seed) -> np.ndarray:
    """Labels observes sous un effet SIMULE de la campagne.

    Controle : vrais labels, inchanges. Traitement : chaque churner reel est
    retenu (label 1 -> 0) avec la probabilite `relative_reduction`, donc le
    churn du groupe baisse de cette fraction en esperance. Aucun non-churner
    ne devient churner (hypothese : la campagne ne nuit a personne).
    """
    if not 0 <= relative_reduction <= 1:
        raise ValueError("relative_reduction doit etre dans [0, 1]")
    y = np.asarray(y).astype(int)
    treated = np.asarray(treated, dtype=bool)
    retained = treated & (y == 1) & (np.random.default_rng(seed).random(len(y)) < relative_reduction)
    y_obs = y.copy()
    y_obs[retained] = 0
    return y_obs


# --- Analyse -----------------------------------------------------------------

@dataclass(frozen=True)
class TwoProportionResult:
    n_control: int
    n_treatment: int
    churn_control: int
    churn_treatment: int
    rate_control: float
    rate_treatment: float
    absolute_diff: float
    relative_diff: float
    ci_low: float
    ci_high: float
    z: float
    p_value: float
    significant: bool

    def as_dict(self) -> dict:
        return asdict(self)


def two_proportion_test(
    churn_control: int, n_control: int, churn_treatment: int, n_treatment: int, alpha: float = 0.05
) -> TwoProportionResult:
    """Test z bilateral de deux proportions (variance poolee sous H0).

    Intervalle de confiance de la difference (traitement - controle) : Wald,
    variance non poolee, niveau 1 - alpha.
    """
    rate_c, rate_t = churn_control / n_control, churn_treatment / n_treatment
    z, p = proportions_ztest([churn_treatment, churn_control], [n_treatment, n_control],
                             alternative="two-sided")
    low, high = confint_proportions_2indep(churn_treatment, n_treatment, churn_control, n_control,
                                           method="wald", compare="diff", alpha=alpha)
    diff = rate_t - rate_c
    return TwoProportionResult(
        n_control=int(n_control), n_treatment=int(n_treatment),
        churn_control=int(churn_control), churn_treatment=int(churn_treatment),
        rate_control=rate_c, rate_treatment=rate_t,
        absolute_diff=diff, relative_diff=diff / rate_c if rate_c > 0 else float("nan"),
        ci_low=float(low), ci_high=float(high), z=float(z), p_value=float(p),
        significant=bool(p < alpha),
    )


def analyze_groups(y, treated, alpha: float = 0.05) -> TwoProportionResult:
    """two_proportion_test a partir des labels et de l'assignation."""
    y = np.asarray(y).astype(int)
    treated = np.asarray(treated, dtype=bool)
    return two_proportion_test(int(y[~treated].sum()), int((~treated).sum()),
                               int(y[treated].sum()), int(treated.sum()), alpha=alpha)



def _z_pvalues(churn_t, n_t, churn_c, n_c) -> tuple[np.ndarray, np.ndarray]:
    """Meme test que two_proportion_test, vectorise sur des tableaux de tirages."""
    churn_t, n_t, churn_c, n_c = (np.asarray(a, dtype=float) for a in (churn_t, n_t, churn_c, n_c))
    pooled = (churn_t + churn_c) / (n_t + n_c)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_t + 1 / n_c))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (churn_t / n_t - churn_c / n_c) / se, 0.0)
    return z, 2 * stats.norm.sf(np.abs(z))


def _simulate(y, relative_reduction: float, n_sims: int, seed, p_treatment: float):
    """n_sims experiences : nouvelle assignation (et nouvel effet simule) a chaque tirage."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    treated = rng.random((n_sims, len(y))) < p_treatment
    churn = np.broadcast_to(y == 1, treated.shape)
    if relative_reduction > 0:
        churn = churn & ~(treated & (rng.random(treated.shape) < relative_reduction))
    n_t = treated.sum(axis=1)
    n_c = len(y) - n_t
    churn_t = (churn & treated).sum(axis=1)
    churn_c = (churn & ~treated).sum(axis=1)
    return _z_pvalues(churn_t, n_t, churn_c, n_c)


def aa_false_positive_rate(y, n_sims: int, seed, alpha: float = 0.05, p_treatment: float = 0.5) -> dict:
    """Test A/A : n_sims assignations sans aucun effet, part des tests significatifs.

    Sans effet, le test doit rejeter H0 dans environ alpha des cas (faux
    positifs). Intervalle de confiance a 95 % de ce taux : Wilson.
    """
    _, p = _simulate(y, 0.0, n_sims, seed, p_treatment)
    k = int((p < alpha).sum())
    ci = stats.binomtest(k, n_sims).proportion_ci(confidence_level=0.95, method="wilson")
    return {"n_sims": n_sims, "false_positives": k, "rate": k / n_sims,
            "ci_low": float(ci.low), "ci_high": float(ci.high)}


def simulated_power(y, relative_reduction: float, n_sims: int, seed, alpha: float = 0.05,
                    p_treatment: float = 0.5) -> float:
    """Puissance observee : part des n_sims experiences simulees qui detectent une BAISSE.

    Compte les tests significatifs (bilateral, p < alpha) dont la difference
    est negative (moins de churn dans le groupe traite).
    """
    z, p = _simulate(y, relative_reduction, n_sims, seed, p_treatment)
    return float(((p < alpha) & (z < 0)).mean())


# --- Garde-fou de cout (hypotheses du projet) --------------------------------

def campaign_cost(n_treated: int, action_cost: float) -> float:
    """Cout de la campagne : clients traites x cout d'une action (hypothese)."""
    return float(n_treated * action_cost)


def expected_net_benefit(n_treated: int, p_control: float, relative_reduction: float,
                         action_cost: float, churn_loss: float) -> dict:
    """Calcul SOUS HYPOTHESES, pas un gain mesure.

    churners evites = n_treated * p_control * relative_reduction
    benefice net    = churners evites * churn_loss - n_treated * action_cost
    rentable si relative_reduction > action_cost / (p_control * churn_loss)
    """
    avoided = n_treated * p_control * relative_reduction
    cost = campaign_cost(n_treated, action_cost)
    return {
        "n_treated": n_treated,
        "avoided_churners": avoided,
        "campaign_cost": cost,
        "avoided_loss": avoided * churn_loss,
        "net_benefit": avoided * churn_loss - cost,
        "break_even_relative_reduction": action_cost / (p_control * churn_loss),
    }
