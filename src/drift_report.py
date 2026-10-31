"""
drift_report.py -- rapport de derive (JSON + Markdown) du modele servi.

Reference : X_train (models/test_split.pkl). Courant : X_test, X_train, ou un
fichier CSV aux colonnes du jeu Telco (customerID et Churn facultatifs). Les
deux jeux sont scores par le modele servi (models/serving_config.json), au
seuil servi. Les seuils d'alerte sont lus dans reports/drift/alert_thresholds.json
s'il existe (seuils figes par le notebook 03), sinon la convention PSI 0,1 / 0,25.

Les metriques sont journalisees dans MLflow en local (file:./mlruns) ; un
echec de MLflow affiche un avertissement et ne bloque jamais le rapport.

Le jeu IBM Telco n'a aucune derive reelle : test contre train ne doit rien
signaler ; la detection est validee sur des derives injectees (notebook 03).

Usage :
    python -m src.drift_report --current test
    python -m src.drift_report --current chemin/vers/clients.csv [--no-mlflow]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from src.drift import (
    Thresholds,
    drift_columns,
    feature_drift,
    overall_level,
    prediction_drift,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT_DIR = ROOT / "reports" / "drift"
THRESHOLDS_FILE = OUT_DIR / "alert_thresholds.json"
DISCLAIMER = ("Jeu public IBM Telco, sans derive reelle connue. La detection a ete validee sur des derives "
              "injectees (notebooks/03_drift_monitoring), pas sur un cas reel. Seuils PSI : convention "
              "d'usage, pas une regle.")
REPORT_KEYS = ("generated_at", "disclaimer", "model", "reference", "current", "thresholds",
               "features", "prediction", "summary")


class DriftInputError(ValueError):
    """Jeu courant inutilisable (fichier absent, colonnes manquantes ou illisibles)."""


def load_thresholds(path: Path = THRESHOLDS_FILE) -> Thresholds:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return Thresholds(**{k: data[k] for k in ("psi_watch", "psi_alert", "alpha")})
    return Thresholds()


def read_current_csv(path, expected_columns: list[str]) -> pd.DataFrame:
    """Lit un CSV courant et verifie qu'il contient les colonnes du modele."""
    path = Path(path)
    if not path.is_file():
        raise DriftInputError(f"fichier courant introuvable : {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # fichier vide, encodage, separateur...
        raise DriftInputError(f"lecture impossible de {path} : {exc}") from exc
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise DriftInputError(f"{path} : colonnes manquantes {missing} ; attendues : {expected_columns}")
    if df.empty:
        raise DriftInputError(f"{path} : aucune ligne")
    df = df[expected_columns].copy()
    # meme conversion que src.preprocessing.clean_data (espaces -> 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    for col in ("tenure", "MonthlyCharges", "SeniorCitizen"):
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            raise DriftInputError(f"{path} : valeurs non numeriques dans {col}")
        df[col] = converted
    return df


def build_report(X_ref: pd.DataFrame, X_cur: pd.DataFrame, p_ref, p_cur, *, threshold: float,
                 thresholds: Thresholds, reference_name: str, current_name: str, model_path: str) -> dict:
    numeric, categorical = drift_columns(X_ref)
    table = feature_drift(X_ref, X_cur, numeric, categorical, thresholds)
    pred = prediction_drift(p_ref, p_cur, threshold, thresholds)
    levels = list(table["level"]) + [pred["level"]]
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
        "model": {"path": model_path, "threshold": threshold},
        "reference": {"name": reference_name, "n": len(X_ref)},
        "current": {"name": current_name, "n": len(X_cur)},
        "thresholds": thresholds.as_dict(),
        "features": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}
                     for row in table.to_dict("records")],
        "prediction": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in pred.items()},
        "summary": {
            "n_features": len(table),
            "n_surveillance": int((table["level"] == "surveillance").sum()),
            "n_alerte": int((table["level"] == "alerte").sum()),
            "flagged": table.loc[table["level"] != "aucun", "variable"].tolist(),
            "overall_level": overall_level(levels),
        },
    }


def to_markdown(report: dict) -> str:
    s, p = report["summary"], report["prediction"]
    lines = [
        "# Rapport de derive",
        "",
        f"> {report['disclaimer']}",
        "",
        f"- Genere le : {report['generated_at']}",
        f"- Modele : `{report['model']['path']}`, seuil {report['model']['threshold']}",
        (f"- Reference : {report['reference']['name']} ({report['reference']['n']} lignes) ; "
        f"courant : {report['current']['name']} ({report['current']['n']} lignes)"),
        (f"- Seuils : PSI surveillance {report['thresholds']['psi_watch']}, alerte "
        f"{report['thresholds']['psi_alert']}, alpha {report['thresholds']['alpha']} (p corrigee BH)"),
        (f"- **Niveau global : {s['overall_level']}** ; variables en surveillance : {s['n_surveillance']}, "
        f"en alerte : {s['n_alerte']} sur {s['n_features']}"),
        "",
        "## Predictions",
        "",
        (f"PSI {p['psi']:.4f}, KS {p['ks_statistic']:.4f} (p = {p['p_value']:.4f}) ; part au-dessus du seuil "
        f"{p['threshold']} : {p['share_above_ref']:.2%} (reference) contre {p['share_above_cur']:.2%} "
        f"(courant) ; niveau : {p['level']}."),
        "",
        "## Variables",
        "",
        "| Variable | Type | PSI | Test | Statistique | p | p corrigee | Niveau |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for f in sorted(report["features"], key=lambda r: -r["psi"]):
        lines.append(f"| {f['variable']} | {f['type']} | {f['psi']:.4f} | {f['test']} | {f['statistic']:.4f} "
                     f"| {f['p_value']:.4f} | {f['p_adjusted']:.4f} | {f['level']} |")
    return "\n".join(lines) + "\n"


def write_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "drift_report.json", out_dir / "drift_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    return json_path, md_path


def log_to_mlflow(report: dict, artifacts: tuple[Path, ...] = (), tracking_uri: str | None = None) -> bool:
    """Journalise les metriques de derive dans MLflow ; ne leve jamais d'exception."""
    try:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        from src.tracking import DEFAULT_TRACKING_URI, EXPERIMENT_NAME

        mlflow.set_tracking_uri(tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name="drift_report"):
            mlflow.log_params({"reference": report["reference"]["name"], "current": report["current"]["name"],
                               **{f"threshold_{k}": v for k, v in report["thresholds"].items()}})
            metrics = {f"psi_{f['variable']}": f["psi"] for f in report["features"]}
            metrics.update({"prediction_psi": report["prediction"]["psi"],
                            "share_above_threshold_cur": report["prediction"]["share_above_cur"],
                            "n_surveillance": report["summary"]["n_surveillance"],
                            "n_alerte": report["summary"]["n_alerte"]})
            mlflow.log_metrics(metrics)
            for path in artifacts:
                mlflow.log_artifact(str(path), artifact_path="drift")
        return True
    except Exception:
        logger.warning("echec de la journalisation MLflow ; le rapport de derive reste valide.", exc_info=True)
        return False


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--current", default="test", help="test, train ou chemin d'un fichier CSV")
    parser.add_argument("--model-dir", default=str(MODELS))
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    config = json.loads((model_dir / "serving_config.json").read_text())
    split = joblib.load(model_dir / "test_split.pkl")
    model_path = ROOT / config["model_path"]
    model = joblib.load(model_path)
    X_ref = split["X_train"]
    if args.current in ("test", "train"):
        X_cur, name = split[f"X_{args.current}"], args.current
    else:
        try:
            X_cur, name = read_current_csv(args.current, list(X_ref.columns)), Path(args.current).name
        except DriftInputError as exc:
            parser.exit(2, f"erreur : {exc}\n")

    report = build_report(X_ref, X_cur, model.predict_proba(X_ref)[:, 1], model.predict_proba(X_cur)[:, 1],
                          threshold=config["threshold"], thresholds=load_thresholds(),
                          reference_name="train", current_name=name, model_path=config["model_path"])
    paths = write_report(report, Path(args.out))
    if not args.no_mlflow:
        log_to_mlflow(report, paths)
    s = report["summary"]
    print(f"Rapport ecrit : {paths[0]} et {paths[1]} ; niveau global {s['overall_level']} "
          f"({s['n_surveillance']} en surveillance, {s['n_alerte']} en alerte sur {s['n_features']} variables ; "
          f"predictions : {report['prediction']['level']})")
    return report


if __name__ == "__main__":
    main()
