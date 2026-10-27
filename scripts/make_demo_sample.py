"""Write web/sample.json from the simulator. The public repo never ships real Wallet rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from selena_ml.data.inject import inject_anomalies
from selena_ml.data.synthetic import simulate_transactions

DEMO_CATEGORIES = [
    "Food & Drinks",
    "Groceries",
    "Transportation",
    "Coffee",
    "Education & development",
]


def main() -> None:
    rng = np.random.default_rng(7)
    df = simulate_transactions(n=60, seed=7)[["date", "amount"]].copy()
    df["category"] = rng.choice(DEMO_CATEGORIES, size=len(df))
    injected, _ = inject_anomalies(df, rate=0.05, seed=11, types=("spike",))

    payload = {
        "transactions": [
            {
                "date": row.date.strftime("%Y-%m-%d"),
                "amount": float(row.amount),
                "category": row.category,
            }
            for row in injected.itertuples()
        ]
    }
    out = Path("web/sample.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"{len(payload['transactions'])} synthetic transactions -> {out}")


if __name__ == "__main__":
    main()
