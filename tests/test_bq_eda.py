import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bq_eda import (  # noqa: E402
    SCHEMA,
    SQL_DIR,
    pandas_reference,
    prepare_for_bq,
    render_sql,
    results_match,
    tenure_band,
)

RAW = ROOT / "data" / "raw" / "telco_churn.csv"


@pytest.fixture(scope="module")
def df():
    return prepare_for_bq(pd.read_csv(RAW))


def test_schema_covers_every_csv_column():
    assert [c for c, _, _ in SCHEMA] == list(pd.read_csv(RAW, nrows=1).columns)


def test_prepare_keeps_blank_total_charges_as_null(df):
    assert df["TotalCharges"].isna().sum() == 11
    assert (df.loc[df["TotalCharges"].isna(), "tenure"] == 0).all()
    assert df["tenure"].dtype == "int64"


def test_prepare_rejects_missing_column():
    with pytest.raises(ValueError):
        prepare_for_bq(pd.DataFrame({"customerID": ["x"]}))


def test_tenure_band_boundaries():
    bands = tenure_band(pd.Series([0, 12, 13, 24, 25, 48, 49, 72]))
    assert list(bands) == ["00-12", "00-12", "13-24", "13-24", "25-48", "25-48", "49-72", "49-72"]


def test_every_sql_file_has_a_pandas_reference_and_table_placeholder(df):
    refs = pandas_reference(df)
    files = sorted((ROOT / SQL_DIR).glob("*.sql"))
    assert {p.stem for p in files} == set(refs)
    for p in files:
        sql = render_sql(p, "proj.ds.tbl")
        assert "`proj.ds.tbl`" in sql and "{table}" not in sql


def test_reference_counts_are_consistent(df):
    for name, table in pandas_reference(df).items():
        assert table["n_clients"].sum() == len(df), name
        assert table["n_churners"].sum() == (df["Churn"] == "Yes").sum(), name


def test_results_match_ignores_row_order(df):
    ref = pandas_reference(df)["01_churn_by_contract"]
    shuffled = ref.sample(frac=1, random_state=0)
    assert results_match(shuffled, ref)
    broken = ref.copy()
    broken.loc[0, "n_churners"] += 1
    assert not results_match(broken, ref)
