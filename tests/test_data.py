from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/churn.csv")
EXPECTED_COLUMNS = {
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "usage_score",
    "late_payments",
    "discount_rate",
    "contract_type",
    "churn",
}
VALID_CONTRACTS = {"monthly", "annual", "two_year"}


def load_data() -> pd.DataFrame:
    assert DATA_PATH.exists(), "Run: python make_data.py"
    return pd.read_csv(DATA_PATH)


def test_expected_columns_exist():
    df = load_data()
    assert set(df.columns) == EXPECTED_COLUMNS


def test_target_is_binary_and_plausible():
    df = load_data()
    assert set(df["churn"].unique()) <= {0, 1}
    assert 0.15 <= df["churn"].mean() <= 0.45


def test_dataset_has_enough_rows():
    df = load_data()
    assert len(df) >= 1000


def test_contract_type_stays_in_known_vocabulary():
    df = load_data()
    assert set(df["contract_type"].dropna().unique()) <= VALID_CONTRACTS
