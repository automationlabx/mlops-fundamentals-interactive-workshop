from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTIONS_PATH = Path("monitoring/predictions.csv")
REFERENCE_PATH = Path("data/reference_stats.json")
DRIFT_THRESHOLD = 0.75
MAX_SHIFT_DISPLAY = 2.5
NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "usage_score",
    "late_payments",
    "discount_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize local prediction traffic and write a visual drift report."
    )
    parser.add_argument(
        "--report",
        default="monitoring/report.html",
        help="Path for the standalone HTML monitoring report.",
    )
    return parser.parse_args()


def calculate_monitoring_summary() -> dict[str, object]:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError("No monitoring data found. Start the API and generate traffic first.")
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError("data/reference_stats.json is missing. Run: python make_data.py")

    predictions = pd.read_csv(PREDICTIONS_PATH)
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

    requests = len(predictions)
    churn_rate = float(predictions["churn"].mean())
    average_probability = float(predictions["churn_probability"].mean())
    p95_latency = float(np.percentile(predictions["latency_ms"], 95))

    shifts: list[tuple[str, float]] = []
    alerts: list[str] = []
    for column in NUMERIC_FEATURES:
        current_mean = float(predictions[column].mean())
        train_mean = float(reference["numeric"][column]["mean"])
        train_std = float(reference["numeric"][column]["std"])
        shift = abs(current_mean - train_mean) / train_std if train_std > 0 else 0.0
        shifts.append((column, shift))
        if shift >= DRIFT_THRESHOLD:
            alerts.append(column)

    return {
        "requests": requests,
        "churn_rate": churn_rate,
        "average_probability": average_probability,
        "p95_latency": p95_latency,
        "shifts": shifts,
        "alerts": alerts,
        "has_alert": bool(alerts),
    }


def write_html_report(report_path: Path, summary: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    requests = int(summary["requests"])
    churn_rate = float(summary["churn_rate"])
    average_probability = float(summary["average_probability"])
    p95_latency = float(summary["p95_latency"])
    shifts = list(summary["shifts"])
    has_alert = bool(summary["has_alert"])
    threshold_pct = min(DRIFT_THRESHOLD / MAX_SHIFT_DISPLAY * 100, 100)

    rows: list[str] = []
    for name, shift in shifts:
        width = min(float(shift) / MAX_SHIFT_DISPLAY * 100, 100)
        state = "alert" if float(shift) >= DRIFT_THRESHOLD else "ok"
        label = "crossed threshold" if state == "alert" else "within threshold"
        rows.append(
            f"""
            <div class="feature-row">
              <div class="feature-head"><span>{html.escape(str(name))}</span><strong>{float(shift):.2f}&sigma;</strong></div>
              <div class="track">
                <div class="threshold" style="left:{threshold_pct:.1f}%"></div>
                <div class="bar {state}" style="width:{width:.1f}%"></div>
              </div>
              <div class="feature-note {state}">{label}</div>
            </div>
            """
        )

    status_class = "alert" if has_alert else "ok"
    status_text = "ALERT: possible drift" if has_alert else "OK: no drift threshold crossed"
    report_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLOps Workshop Monitoring Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 0; background: #f4f7f9; color: #17212b; }}
  main {{ max-width: 1000px; margin: 36px auto; padding: 0 22px 36px; }}
  h1 {{ margin-bottom: 6px; }}
  .sub {{ color: #5b6875; margin-top: 0; }}
  .status {{ margin: 22px 0; padding: 14px 18px; border-radius: 10px; font-weight: 700; }}
  .status.ok {{ background: #e8f7ee; color: #176b3a; }}
  .status.alert {{ background: #fdeaea; color: #a72828; }}
  .kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 26px; }}
  .card {{ background: white; border: 1px solid #dbe3e8; border-radius: 10px; padding: 16px; }}
  .card span {{ display: block; color: #667481; font-size: 13px; margin-bottom: 8px; }}
  .card strong {{ font-size: 24px; }}
  .panel {{ background: white; border: 1px solid #dbe3e8; border-radius: 10px; padding: 20px; }}
  .feature-row {{ margin: 18px 0; }}
  .feature-head {{ display: flex; justify-content: space-between; margin-bottom: 7px; }}
  .track {{ position: relative; height: 15px; border-radius: 8px; background: #e8edf1; overflow: hidden; }}
  .bar {{ height: 100%; border-radius: 8px; }}
  .bar.ok {{ background: #4c9f70; }}
  .bar.alert {{ background: #d9534f; }}
  .threshold {{ position: absolute; top: 0; bottom: 0; width: 2px; background: #202a34; z-index: 2; opacity: .8; }}
  .feature-note {{ font-size: 12px; margin-top: 5px; }}
  .feature-note.ok {{ color: #46765a; }}
  .feature-note.alert {{ color: #a72828; font-weight: 700; }}
  .legend {{ color: #667481; font-size: 13px; margin-top: 18px; }}
  @media (max-width: 760px) {{ .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
</style>
</head>
<body>
<main>
  <h1>MLOps Workshop Monitoring Report</h1>
  <p class="sub">Current traffic compared with the deterministic training reference.</p>
  <div class="status {status_class}">{status_text}</div>
  <section class="kpis">
    <div class="card"><span>Requests</span><strong>{requests}</strong></div>
    <div class="card"><span>Predicted churn rate</span><strong>{churn_rate:.3f}</strong></div>
    <div class="card"><span>Average probability</span><strong>{average_probability:.3f}</strong></div>
    <div class="card"><span>p95 latency</span><strong>{p95_latency:.2f} ms</strong></div>
  </section>
  <section class="panel">
    <h2>Feature mean shift</h2>
    <p class="sub">Absolute shift in training standard deviations (&sigma;).</p>
    {''.join(rows)}
    <p class="legend">The vertical marker is the workshop alert threshold: {DRIFT_THRESHOLD:.2f}&sigma;. Bar scale is capped at {MAX_SHIFT_DISPLAY:.1f}&sigma; for readability.</p>
  </section>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    try:
        summary = calculate_monitoring_summary()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Requests: {summary['requests']}")
    print(f"Predicted churn rate: {float(summary['churn_rate']):.3f}")
    print(f"Average probability: {float(summary['average_probability']):.3f}")
    print(f"p95 latency (ms): {float(summary['p95_latency']):.2f}")
    print("Drift proxy (absolute mean shift in training standard deviations):")

    for column, shift in summary["shifts"]:
        print(f"- {column}: {float(shift):.2f}")

    alerts = list(summary["alerts"])
    if alerts:
        print(f"ALERT: possible drift - {', '.join(str(name) for name in alerts)}")
    else:
        print("OK: no numeric feature crossed the workshop drift threshold")

    report_path = Path(parse_args().report)
    write_html_report(report_path, summary)
    print(f"Visual report: {report_path.as_posix()}")


if __name__ == "__main__":
    main()
