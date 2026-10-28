"""
interpret.py -- interpretabilite du modele retenu (logistic_regression) :
importance des features via les coefficients standardises, et valeurs
SHAP pour expliquer les predictions individuelles.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


def get_feature_names(pipeline) -> list[str]:
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())


def plot_coefficients(pipeline, feature_names, fig_dir: Path, top_n: int = 15):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        print(f"{type(model).__name__} n'a pas de coefficients lineaires -- "
              f"utiliser feature_importances_ ou SHAP a la place.")
        return

    coefs = pd.Series(model.coef_[0], index=feature_names).sort_values()
    top = pd.concat([coefs.head(top_n // 2), coefs.tail(top_n // 2)])

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in top.values]
    top.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Coefficients de la regression logistique\n(rouge = augmente le risque de churn)")
    ax.set_xlabel("Coefficient (sur variables standardisees)")
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_coefficients.png", dpi=120)
    plt.close(fig)
    return coefs


def plot_shap(pipeline, X_test, feature_names, fig_dir: Path, sample_size: int = 300):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    X_transformed = preprocessor.transform(X_sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    explainer = shap.LinearExplainer(model, X_transformed)
    shap_values = explainer(X_transformed)
    shap_values.feature_names = feature_names

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--fig-dir", default="reports/figures")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pipeline = joblib.load(model_dir / "best_model.pkl")
    split = joblib.load(model_dir / "test_split.pkl")
    X_test = split["X_test"]

    feature_names = get_feature_names(pipeline)

    coefs = plot_coefficients(pipeline, feature_names, fig_dir)
    if coefs is not None:
        print("Top 5 features augmentant le risque de churn :")
        print(coefs.sort_values(ascending=False).head(5))
        print("\nTop 5 features reduisant le risque de churn :")
        print(coefs.sort_values().head(5))

    plot_shap(pipeline, X_test, feature_names, fig_dir)
    print(f"\nFigures d'interpretabilite sauvegardees dans {fig_dir}/")


if __name__ == "__main__":
    main()
