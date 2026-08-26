# MLOps Fundamentals Workshop

A local-first hands-on project for practising the core MLOps lifecycle without Docker or cloud deployment.

The workshop covers:

- deterministic data generation and validation;
- experiment tracking and a local Model Registry with MLflow;
- baseline and candidate model comparison;
- an explicit model-quality gate;
- champion promotion and rollback;
- local FastAPI serving with the active champion version visible in /health;
- a live browser dashboard that shows serving/registry mismatch, candidate gate status, and current monitoring state;
- data, model, and API tests;
- a local quality command;
- prediction logging, a simple drift signal, and a standalone visual monitoring report;
- final independent verification with GitHub Actions.

## Local setup

Use the Workshop Guide and work through the exercises in order.

```text
python -m venv .venv
```

Activate the environment, then install dependencies:

```text
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Main local commands

```text
python make_data.py
python -m pytest tests/test_data.py -v
python train.py --kind logreg --run-name baseline
python train.py --kind rf --run-name rf_candidate --max-depth 6 --n-estimators 200 --candidate
python promote.py
uvicorn serve:app --reload --port 8000
# open http://127.0.0.1:8000/dashboard
python -m pytest -v
python run_checks.py
```

The rollback, failure, monitoring, Git hook, and GitHub Actions exercises are described in `Workshop_Guide.docx`.

## Participant tasks

During the workshop you will add one API validation test in Part 5 and create the GitHub Actions verification workflow in Part 10. The starter intentionally does not include those completed student edits.
