from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "churn-workshop-model"
SERVING_MODEL_PATH = Path("models/champion_model.joblib")
TEMP_MODEL_PATH = Path("models/.champion_model.joblib.tmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move champion to an older registered model version.")
    parser.add_argument("--version", required=True, help="Existing registered model version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = str(args.version)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    try:
        client.get_model_version(MODEL_NAME, version)
    except Exception as exc:
        raise SystemExit(f"Model version {version} does not exist: {exc}") from exc

    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    except Exception as exc:
        raise SystemExit(f"Model version {version} could not be loaded: {exc}") from exc

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

    print(f"ROLLBACK - champion -> version {version}")
    print(f"Serving copy: {SERVING_MODEL_PATH.as_posix()}")


if __name__ == "__main__":
    main()
