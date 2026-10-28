"""
preprocessing.py -- chargement et nettoyage du dataset Telco Customer Churn,
et construction du pipeline scikit-learn de pretraitement.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COL = "customerID"

# SeniorCitizen est deja encode en 0/1 dans le fichier source -> traite comme numerique.
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le dataframe brut Telco Churn.

    - TotalCharges est lu comme texte car 11 lignes contiennent un espace
      (' ') au lieu d'un nombre : ce sont des clients avec tenure == 0,
      donc un total facture nul -> on convertit en numerique et on remplit
      ces 11 valeurs par 0 plutot que de les supprimer (peu de lignes, et le
      motif metier -- nouveau client, rien facture -- est clair).
    - customerID est un identifiant sans valeur predictive -> retire.
    - Churn (Yes/No) est encode en 0/1 pour la modelisation.
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = df.drop(columns=[ID_COL])

    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Retourne (colonnes_numeriques, colonnes_categorielles) presentes dans df,
    hors colonne cible."""
    categorical = [
        c for c in df.columns
        if c not in NUMERIC_FEATURES and c != TARGET
    ]
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    return numeric, categorical


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Construit le ColumnTransformer : standardisation des variables
    numeriques (utile pour la regression logistique, neutre pour les
    modeles a base d'arbres) + one-hot encoding des variables categorielles.
    """
    numeric, categorical = get_feature_columns(df)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical),
        ]
    )


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y
