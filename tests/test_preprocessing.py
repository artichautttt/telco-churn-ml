import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import clean_data, get_feature_columns, split_X_y


def make_raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "customerID": ["0001-AAA", "0002-BBB", "0003-CCC"],
        "gender": ["Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "tenure": [0, 12, 24],
        "MonthlyCharges": [29.85, 56.95, 42.30],
        "TotalCharges": [" ", "683.4", "1015.2"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "Churn": ["No", "Yes", "No"],
    })


def test_total_charges_blank_becomes_zero():
    df = clean_data(make_raw_df())
    assert df.loc[0, "TotalCharges"] == 0.0


def test_total_charges_is_numeric():
    df = clean_data(make_raw_df())
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])


def test_customer_id_is_dropped():
    df = clean_data(make_raw_df())
    assert "customerID" not in df.columns


def test_churn_encoded_as_binary():
    df = clean_data(make_raw_df())
    assert set(df["Churn"].unique()) <= {0, 1}
    assert df.loc[1, "Churn"] == 1
    assert df.loc[0, "Churn"] == 0


def test_get_feature_columns_splits_numeric_and_categorical():
    df = clean_data(make_raw_df())
    numeric, categorical = get_feature_columns(df)
    assert "tenure" in numeric
    assert "MonthlyCharges" in numeric
    assert "TotalCharges" in numeric
    assert "Contract" in categorical
    assert "Churn" not in numeric and "Churn" not in categorical


def test_split_X_y_removes_target_from_X():
    df = clean_data(make_raw_df())
    X, y = split_X_y(df)
    assert "Churn" not in X.columns
    assert y.name == "Churn"
    assert len(X) == len(y) == 3
