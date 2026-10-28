"""
bq_eda.py -- analyse exploratoire en SQL sur BigQuery.

1. charge data/raw/telco_churn.csv dans BigQuery (schema explicite, table
   remplacee a chaque chargement) ;
2. execute chaque requete de sql/*.sql ;
3. exporte les resultats en CSV dans reports/sql/ ;
4. recoupe chaque resultat avec le meme calcul fait en pandas sur le CSV
   local (garde-fou contre une erreur de chargement ou de SQL).

Authentification : Application Default Credentials
(`gcloud auth application-default login`). Compatible BigQuery Sandbox
(chargement par job, pas de streaming insert).

Usage :
    python -m src.bq_eda --project MON_PROJET            # cree le dataset telco_churn si besoin
    python -m src.bq_eda --project MON_PROJET --dataset MON_DATASET
        # ecrit dans un dataset EXISTANT sans tenter de le creer : seuls les
        # droits de creer une table et de lancer des jobs sont necessaires
        # (ex. roles BigQuery Data Editor sur le dataset + Job User sur le projet)
    python -m src.bq_eda --project MON_PROJET --dataset MON_DATASET --skip-load
Projet par defaut : variable BQ_PROJECT, sinon le projet des ADC.
Les fichiers SQL utilisent le placeholder {table} (nom complet de la table).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

SQL_DIR = Path("sql")
OUT_DIR = Path("reports/sql")
DEFAULT_DATASET = "telco_churn"
DEFAULT_TABLE = "customers"
DEFAULT_LOCATION = "EU"

# Schema explicite : (colonne, type BigQuery, mode). TotalCharges est NULL
# pour les 11 clients a tenure 0 (valeur vide dans le CSV source).
SCHEMA = [
    ("customerID", "STRING", "REQUIRED"),
    ("gender", "STRING", "REQUIRED"),
    ("SeniorCitizen", "INT64", "REQUIRED"),
    ("Partner", "STRING", "REQUIRED"),
    ("Dependents", "STRING", "REQUIRED"),
    ("tenure", "INT64", "REQUIRED"),
    ("PhoneService", "STRING", "REQUIRED"),
    ("MultipleLines", "STRING", "REQUIRED"),
    ("InternetService", "STRING", "REQUIRED"),
    ("OnlineSecurity", "STRING", "REQUIRED"),
    ("OnlineBackup", "STRING", "REQUIRED"),
    ("DeviceProtection", "STRING", "REQUIRED"),
    ("TechSupport", "STRING", "REQUIRED"),
    ("StreamingTV", "STRING", "REQUIRED"),
    ("StreamingMovies", "STRING", "REQUIRED"),
    ("Contract", "STRING", "REQUIRED"),
    ("PaperlessBilling", "STRING", "REQUIRED"),
    ("PaymentMethod", "STRING", "REQUIRED"),
    ("MonthlyCharges", "FLOAT64", "REQUIRED"),
    ("TotalCharges", "FLOAT64", "NULLABLE"),
    ("Churn", "STRING", "REQUIRED"),
]

TENURE_BINS = [-1, 12, 24, 48, np.inf]
TENURE_LABELS = ["00-12", "13-24", "25-48", "49-72"]


def prepare_for_bq(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aligne le CSV brut sur SCHEMA (ordre des colonnes et types)."""
    cols = [c for c, _, _ in SCHEMA]
    missing = set(cols) - set(df_raw.columns)
    if missing:
        raise ValueError(f"colonnes absentes : {sorted(missing)}")
    df = df_raw[cols].copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    for c, typ, _ in SCHEMA:
        if typ == "INT64":
            df[c] = df[c].astype("int64")
        elif typ == "FLOAT64":
            df[c] = df[c].astype("float64")
        else:
            df[c] = df[c].astype(str)
    return df


def tenure_band(tenure: pd.Series) -> pd.Series:
    """Meme decoupage que le CASE WHEN des fichiers SQL."""
    return pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)


def _rate_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = df.assign(churn=(df["Churn"] == "Yes").astype(int)).groupby(keys, observed=True)["churn"]
    out = g.agg(n_clients="size", n_churners="sum").reset_index()
    out["churn_rate"] = (out["n_churners"] / out["n_clients"]).round(4)
    return out


def pandas_reference(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Equivalent pandas de chaque requete SQL (cle = nom du fichier sans .sql),
    colonnes renommees comme dans le SQL."""
    d = df.rename(columns={
        "Contract": "contract",
        "InternetService": "internet_service",
        "PaymentMethod": "payment_method",
    }).assign(tenure_band=lambda x: tenure_band(x["tenure"]))
    return {
        "01_churn_by_contract": _rate_table(d, ["contract"]),
        "02_churn_by_tenure_band": _rate_table(d, ["tenure_band"]),
        "03_churn_by_internet_service": _rate_table(d, ["internet_service"]),
        "04_churn_by_payment_method": _rate_table(d, ["payment_method"]),
        "05_churn_contract_x_tenure": _rate_table(d, ["contract", "tenure_band"]),
    }


def render_sql(path: Path, table: str) -> str:
    return path.read_text(encoding="utf-8").replace("{table}", table)


def results_match(bq: pd.DataFrame, ref: pd.DataFrame) -> bool:
    """Compare deux tables de resultats independamment de l'ordre des lignes."""
    if set(bq.columns) != set(ref.columns) or len(bq) != len(ref):
        return False
    keys = [c for c in ref.columns if c not in ("n_clients", "n_churners", "churn_rate")]
    a = bq[ref.columns].sort_values(keys).reset_index(drop=True)
    b = ref.sort_values(keys).reset_index(drop=True)
    return (
        a[keys].astype(str).equals(b[keys].astype(str))
        and np.array_equal(a["n_clients"].astype(int), b["n_clients"].astype(int))
        and np.array_equal(a["n_churners"].astype(int), b["n_churners"].astype(int))
        and np.allclose(a["churn_rate"].astype(float), b["churn_rate"].astype(float), atol=1e-4)
    )


def load_table(client, df: pd.DataFrame, table_id: str, location: str,
               create_dataset: bool = True) -> None:
    """Charge df dans table_id (remplacee). create_dataset=False : le dataset
    doit deja exister, aucun appel de creation n'est fait."""
    from google.cloud import bigquery

    if create_dataset:
        ds = bigquery.Dataset(table_id.rsplit(".", 1)[0])
        ds.location = location
        client.create_dataset(ds, exists_ok=True)
    job_config = bigquery.LoadJobConfig(
        schema=[bigquery.SchemaField(c, t, mode=m) for c, t, m in SCHEMA],
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
    n = client.get_table(table_id).num_rows
    print(f"Table {table_id} chargee : {n} lignes")
    if n != len(df):
        raise RuntimeError(f"{n} lignes chargees au lieu de {len(df)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/telco_churn.csv")
    parser.add_argument("--project", default=os.environ.get("BQ_PROJECT"))
    parser.add_argument("--dataset", default=None,
                        help="dataset EXISTANT ou ecrire (pas de creation). "
                             f"Absent : '{DEFAULT_DATASET}', cree si besoin.")
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--skip-load", action="store_true")
    args = parser.parse_args()

    from google.cloud import bigquery

    client = bigquery.Client(project=args.project, location=args.location)
    dataset = args.dataset or DEFAULT_DATASET
    table_id = f"{client.project}.{dataset}.{args.table}"

    df = prepare_for_bq(pd.read_csv(args.data))
    if not args.skip_load:
        load_table(client, df, table_id, args.location, create_dataset=args.dataset is None)

    refs = pandas_reference(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for path in sorted(SQL_DIR.glob("*.sql")):
        result = client.query(render_sql(path, table_id)).to_dataframe()
        result.to_csv(OUT_DIR / f"{path.stem}.csv", index=False)
        ok = results_match(result, refs[path.stem]) if path.stem in refs else None
        all_ok &= ok is not False
        print(f"\n== {path.stem}  (recoupement pandas : {ok})")
        print(result.to_string(index=False))
    print(f"\nResultats exportes dans {OUT_DIR}/")
    if not all_ok:
        raise SystemExit("au moins un resultat BigQuery differe du calcul pandas")


if __name__ == "__main__":
    main()
