from fastapi.testclient import TestClient

from serve import app

client = TestClient(app)

VALID = {
    "tenure_months": 12,
    "monthly_spend": 95,
    "support_tickets": 3,
    "usage_score": 55,
    "late_payments": 1,
    "discount_rate": 0.10,
    "contract_type": "monthly",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model"] == "champion"
    assert payload["version"].isdigit()


def test_valid_prediction():
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["churn"], bool)
    assert 0.0 <= payload["churn_probability"] <= 1.0
    assert payload["model"] == "champion"


def test_unknown_contract_type_is_rejected():
    bad = {**VALID, "contract_type": "forever"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


# Part 5.1 student task:
# Add test_spend_out_of_range_is_rejected() here, then re-run the API tests.
