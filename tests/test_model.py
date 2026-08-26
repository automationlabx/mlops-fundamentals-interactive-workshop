from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/champion_model.joblib")


def test_champion_model_loads_and_returns_probability():
    assert MODEL_PATH.exists(), "Promote a passing candidate first."
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame(
        [
            {
                "tenure_months": 12,
                "monthly_spend": 95,
                "support_tickets": 3,
                "usage_score": 55,
                "late_payments": 1,
                "discount_rate": 0.10,
                "contract_type": "monthly",
            }
        ]
    )
    probability = float(model.predict_proba(sample)[0, 1])
    assert 0.0 <= probability <= 1.0
