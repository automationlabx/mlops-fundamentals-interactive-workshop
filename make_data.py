from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROWS = 1500
DATA_PATH = Path("data/churn.csv")
REFERENCE_PATH = Path("data/reference_stats.json")
NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "usage_score",
    "late_payments",
    "discount_rate",
]


def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    tenure = rng.integers(1, 73, ROWS)
    monthly = np.round(np.clip(rng.normal(72, 25, ROWS), 20, 160), 2)
    support = np.clip(rng.poisson(1.6, ROWS), 0, 8)
    usage = np.round(np.clip(rng.normal(62, 20, ROWS), 0, 100), 2)
    late = np.clip(rng.poisson(0.7, ROWS), 0, 5)
    discount = np.round(
        rng.choice(
            [0.00, 0.05, 0.10, 0.15, 0.20, 0.25],
            size=ROWS,
            p=[0.30, 0.18, 0.22, 0.14, 0.10, 0.06],
        ),
        2,
    )
    contract = rng.choice(
        ["monthly", "annual", "two_year"],
        size=ROWS,
        p=[0.55, 0.30, 0.15],
    )

    # The target is intentionally driven mostly by interactions. This gives
    # Logistic Regression a useful baseline while letting Random Forest capture
    # stronger nonlinear patterns during the workshop comparison.
    risk = (
        2.0 * ((support >= 3) & (usage < 55))
        + 2.0 * ((late >= 2) & (monthly > 85))
        + 1.8 * ((tenure < 15) & (contract == "monthly"))
        + 1.6 * ((monthly > 105) & (discount >= 0.15))
        + 1.5 * ((support == 0) & (usage < 30))
        + 1.2 * ((tenure > 50) & (monthly > 100))
        - 1.5 * ((tenure > 48) & (contract == "two_year"))
        + 0.4 * rng.normal(size=ROWS)
    )

    threshold = np.quantile(risk, 1 - 0.287)
    churn = (risk > threshold).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_spend": monthly,
            "support_tickets": support,
            "usage_score": usage,
            "late_payments": late,
            "discount_rate": discount,
            "contract_type": contract,
            "churn": churn,
        }
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_reference_stats(df: pd.DataFrame, sha256: str) -> None:
    stats = {
        "rows": int(len(df)),
        "target_rate": float(df["churn"].mean()),
        "data_sha256": sha256,
        "numeric": {
            column: {
                "mean": float(df[column].mean()),
                "std": float(df[column].std(ddof=0)),
            }
            for column in NUMERIC_FEATURES
        },
        "contract_type_distribution": {
            str(key): float(value)
            for key, value in df["contract_type"].value_counts(normalize=True).sort_index().items()
        },
    }
    REFERENCE_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = build_dataset()
    # Write values with explicit formatting instead of relying on pandas CSV
    # formatting. This keeps the fingerprint stable across supported platforms.
    columns = [
        "tenure_months",
        "monthly_spend",
        "support_tickets",
        "usage_score",
        "late_payments",
        "discount_rate",
        "contract_type",
        "churn",
    ]
    with DATA_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in df.itertuples(index=False):
            writer.writerow(
                [
                    int(row.tenure_months),
                    f"{row.monthly_spend:.2f}",
                    int(row.support_tickets),
                    f"{row.usage_score:.2f}",
                    int(row.late_payments),
                    f"{row.discount_rate:.2f}",
                    row.contract_type,
                    int(row.churn),
                ]
            )
    sha256 = sha256_file(DATA_PATH)
    write_reference_stats(df, sha256)

    print(f"Created {DATA_PATH.as_posix()} with {len(df)} rows")
    print(f"Target churn rate: {df['churn'].mean():.3f}")
    print(f"SHA-256: {sha256}")
    print(f"Reference stats: {REFERENCE_PATH.as_posix()}")


if __name__ == "__main__":
    main()
