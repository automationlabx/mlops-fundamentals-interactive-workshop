from __future__ import annotations

import argparse
import time

import httpx
import numpy as np

URL = "http://127.0.0.1:8000/predict"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local API traffic for monitoring exercises.")
    parser.add_argument("--profile", choices=["normal", "shifted"], required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--delay-ms", type=int, default=0, help="Optional delay between requests so changes are visible in the live dashboard.")
    return parser.parse_args()


def normal_customer(rng: np.random.Generator) -> dict[str, object]:
    return {
        "tenure_months": int(rng.integers(1, 73)),
        "monthly_spend": round(float(np.clip(rng.normal(72, 25), 20, 160)), 2),
        "support_tickets": int(np.clip(rng.poisson(1.6), 0, 8)),
        "usage_score": round(float(np.clip(rng.normal(62, 20), 0, 100)), 2),
        "late_payments": int(np.clip(rng.poisson(0.7), 0, 5)),
        "discount_rate": float(rng.choice([0.00, 0.05, 0.10, 0.15, 0.20, 0.25], p=[0.30, 0.18, 0.22, 0.14, 0.10, 0.06])),
        "contract_type": str(rng.choice(["monthly", "annual", "two_year"], p=[0.55, 0.30, 0.15])),
    }


def shifted_customer(rng: np.random.Generator) -> dict[str, object]:
    return {
        "tenure_months": int(rng.integers(1, 20)),
        "monthly_spend": round(float(np.clip(rng.normal(125, 15), 70, 180)), 2),
        "support_tickets": int(np.clip(rng.poisson(4.2), 0, 12)),
        "usage_score": round(float(np.clip(rng.normal(43, 14), 0, 100)), 2),
        "late_payments": int(np.clip(rng.poisson(2.1), 0, 8)),
        "discount_rate": float(rng.choice([0.10, 0.15, 0.20, 0.25], p=[0.10, 0.20, 0.35, 0.35])),
        "contract_type": str(rng.choice(["monthly", "annual", "two_year"], p=[0.82, 0.14, 0.04])),
    }


def main() -> None:
    args = parse_args()
    if args.requests < 1:
        raise SystemExit("--requests must be at least 1")
    if args.delay_ms < 0:
        raise SystemExit("--delay-ms must be zero or positive")

    rng = np.random.default_rng(123 if args.profile == "normal" else 456)
    factory = normal_customer if args.profile == "normal" else shifted_customer

    with httpx.Client(timeout=10.0) as client:
        for index in range(args.requests):
            response = client.post(URL, json=factory(rng))
            response.raise_for_status()
            if args.delay_ms:
                time.sleep(args.delay_ms / 1000)
            if (index + 1) % 20 == 0 or index + 1 == args.requests:
                print(f"Sent {index + 1}/{args.requests} {args.profile} requests")


if __name__ == "__main__":
    main()
