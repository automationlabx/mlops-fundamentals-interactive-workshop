from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "local-churn-workshop"
MODEL_NAME = "churn-workshop-model"
DATA_PATH = Path("data/churn.csv")
CANDIDATE_PATH = Path("candidate.json")
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "usage_score",
    "late_payments",
    "discount_rate",
]
CATEGORICAL_FEATURES = ["contract_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and register a workshop churn model.")
    parser.add_argument("--kind", choices=["logreg", "rf", "weak_rf"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="Mark this trained model as the current promotion candidate.",
    )
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    if args.kind == "logreg":
        estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    elif args.kind == "rf":
        estimator = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=20,
            max_depth=1,
            class_weight={0: 2, 1: 1},
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def main() -> None:
    args = parse_args()
    if not DATA_PATH.exists():
        raise SystemExit("data/churn.csv is missing. Run: python make_data.py")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model(args)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    data_sha256 = file_sha256(DATA_PATH)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    params = {
        "kind": args.kind,
        "random_state": RANDOM_STATE,
        "test_size": 0.25,
        "data_sha256": data_sha256,
        "role": "candidate" if args.candidate else "reference",
    }
    if args.kind == "rf":
        params.update({"max_depth": args.max_depth, "n_estimators": args.n_estimators})
    elif args.kind == "weak_rf":
        params.update({"max_depth": 1, "n_estimators": 20, "class_weight": "0:2,1:1"})

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        signature_input = X_train.copy()
        integer_columns = signature_input.select_dtypes(include="integer").columns
        signature_input[integer_columns] = signature_input[integer_columns].astype("float64")
        signature = infer_signature(signature_input, model.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=signature_input.head(3),
            registered_model_name=MODEL_NAME,
        )
        run_id = run.info.run_id

    if model_info.registered_model_version is None:
        raise SystemExit("MLflow did not return a registered model version.")
    version = str(model_info.registered_model_version)

    if args.candidate:
        client = mlflow.MlflowClient(tracking_uri=TRACKING_URI)
        client.set_registered_model_alias(MODEL_NAME, "candidate", version)
        candidate = {
            "model_name": MODEL_NAME,
            "version": version,
            "run_id": run_id,
            "run_name": args.run_name,
            "kind": args.kind,
            "data_sha256": data_sha256,
            "metrics": metrics,
        }
        CANDIDATE_PATH.write_text(json.dumps(candidate, indent=2), encoding="utf-8")

    print(f"Run: {args.run_name}")
    print(f"Kind: {args.kind}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"Run ID: {run_id}")
    print(f"Registered model: {MODEL_NAME} version {version}")
    if args.candidate:
        print("Role: promotion candidate")
        print(f"Alias candidate -> version {version}")
        print(f"Candidate evidence: {CANDIDATE_PATH.as_posix()}")
    else:
        print("Role: reference model (not a promotion candidate)")


if __name__ == "__main__":
    main()
