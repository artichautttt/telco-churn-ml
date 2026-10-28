"""
shap_utils.py -- outils SHAP partages (segmentation, interactions, robustesse).

Le pipeline XGBoost applique un one-hot encoding : une variable d'origine
(ex. Contract) devient plusieurs colonnes. Les valeurs SHAP etant additives,
l'effet d'une variable d'origine est la somme des SHAP de ses colonnes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def feature_origin(preprocessor) -> list[str]:
    """Pour chaque colonne transformee, le nom de la variable d'origine."""
    origin: list[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder" or trans == "drop":
            continue
        if hasattr(trans, "categories_"):
            drop_idx = getattr(trans, "drop_idx_", None)
            for i, (col, cats) in enumerate(zip(cols, trans.categories_)):
                n = len(cats) - (1 if drop_idx is not None and drop_idx[i] is not None else 0)
                origin.extend([col] * n)
        else:
            origin.extend(cols)
    return origin


def transform(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    pre = pipeline.named_steps["preprocessor"]
    Xt = pre.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    return np.asarray(Xt, dtype=float), names


def tree_shap(pipeline, X: pd.DataFrame):
    """Valeurs SHAP (log-odds) du modele a arbres du pipeline.

    Retourne (shap_values (n, p), X_transforme, noms_colonnes, explainer)."""
    import shap

    Xt, names = transform(pipeline, X)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    values = np.asarray(explainer.shap_values(Xt))
    return values, Xt, names, explainer


def aggregate_by_feature(values: np.ndarray, origin: list[str]) -> pd.DataFrame:
    """Somme les SHAP des colonnes one-hot d'une meme variable d'origine."""
    if values.shape[1] != len(origin):
        raise ValueError("nombre de colonnes SHAP different de la longueur de origin")
    df = pd.DataFrame(values, columns=origin)
    return df.T.groupby(level=0, sort=False).sum().T


def mean_abs_importance(values: np.ndarray, origin: list[str]) -> pd.Series:
    """Importance globale par variable d'origine : moyenne de |SHAP agrege|."""
    return aggregate_by_feature(values, origin).abs().mean().sort_values(ascending=False)
