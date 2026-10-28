"""Write web/sample.json from the simulator. The public repo never ships real Wallet rows.

Categories are drawn with the same weights as the real series (top five), because the detector
carries a category-relative feature: sampling categories uniformly makes every row look unusual
for its category, which produced a demo with a 32% flag rate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from selena_ml.data.inject import inject_anomalies
from selena_ml.data.load import load_transactions
from selena_ml.data.synthetic import simulate_transactions

DEMO_CATEGORIES = [
    "Restaurants & fast food",
    "Coffee",
    "Groceries",
    "Transportation",
    "Education & development",
]


def category_weights(n: int) -> np.ndarray:
    """Real top-five category shares, falling back to uniform if the local export is absent."""
    try:
        df = load_transactions("data/processed/spending.parquet")
    except FileNotFoundError:
        return np.full(n, 1 / n)
    counts = df["category"].value_counts()
    weights = np.array([counts.get(c, 0) for c in DEMO_CATEGORIES], dtype=float)
    if weights.sum() == 0:
        return np.full(n, 1 / n)
    return weights / weights.sum()


def window(df: pd.DataFrame) -> pd.DataFrame:
    """Move the synthetic dates into the period the model was trained on.

    The features include month and day-of-month, so a demo set dated January would sit outside
    the training range and every row would score as an outlier. Spacing is preserved.
    """
    try:
        real = load_transactions("data/processed/spending.parquet")
    except FileNotFoundError:
        return df
    # Shift, never stretch: stretching 60 rows across the full 90-day window halves the
    # transaction cadence, and days_since_prev then flags nearly every row.
    offset = (real["date"].max() - pd.Timedelta(days=1)) - df["date"].max()
    df = df.copy()
    df["date"] = df["date"] + offset
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    rng = np.random.default_rng(7)
    df = window(simulate_transactions(n=60, seed=7)[["date", "amount"]].copy())
    weights = category_weights(len(DEMO_CATEGORIES))
    df["category"] = rng.choice(DEMO_CATEGORIES, size=len(df), p=weights)
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
