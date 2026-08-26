from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "churn-workshop-model"
CANDIDATE_PATH = Path("candidate.json")
SERVING_MODEL_PATH = Path("models/champion_model.joblib")
TEMP_MODEL_PATH = Path("models/.champion_model.joblib.tmp")

THRESHOLDS = {
    "accuracy": 0.82,
    "recall": 0.70,
    "roc_auc": 0.85,
}


def main() -> None:
    if not CANDIDATE_PATH.exists():
        raise SystemExit("candidate.json is missing. Train a model with --candidate first.")

    candidate = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    version = str(candidate["version"])
    metrics = candidate["metrics"]

    failures = [
        f"{name}={metrics[name]:.3f} < {minimum:.2f}"
        for name, minimum in THRESHOLDS.items()
        if metrics[name] < minimum
    ]

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    try:
        current_candidate = client.get_model_version_by_alias(MODEL_NAME, "candidate")
    except Exception as exc:
        raise SystemExit(f"Candidate alias is missing in MLflow Registry: {exc}") from exc

    if str(current_candidate.version) != version:
        raise SystemExit(
            "candidate.json does not match the current candidate alias. "
            "Run the intended training command again."
        )

    if failures:
        print(f"FAIL - candidate version {version} was not promoted")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    # Validate and stage the serving copy before changing the champion alias.
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    except Exception as exc:
        raise SystemExit(f"Candidate model version {version} could not be loaded: {exc}") from exc

    SERVING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMP_MODEL_PATH.unlink(missing_ok=True)
    joblib.dump(model, TEMP_MODEL_PATH)

    try:
        previous_champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
        previous_version = str(previous_champion.version)
    except Exception:
        previous_version = None

    try:
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
        os.replace(TEMP_MODEL_PATH, SERVING_MODEL_PATH)
    except Exception:
        if previous_version is None:
            try:
                client.delete_registered_model_alias(MODEL_NAME, "champion")
            except Exception:
                pass
        else:
            try:
                client.set_registered_model_alias(MODEL_NAME, "champion", previous_version)
            except Exception:
                pass
        TEMP_MODEL_PATH.unlink(missing_ok=True)
        raise

    print(f"PASS - version {version} promoted to champion")
    print(f"Serving copy: {SERVING_MODEL_PATH.as_posix()}")


if __name__ == "__main__":
    main()
