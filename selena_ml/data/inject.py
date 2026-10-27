"""Inject synthetic anomalies so an unlabelled series can be evaluated.

Mutations are in place (no rows added or removed) so row -> label stays 1:1.
Four types: spike, duplicate, round_burst, drip (card-testing pattern).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANOMALY_TYPES = ("spike", "duplicate", "round_burst", "drip")


def inject_anomalies(
    df: pd.DataFrame,
    rate: float = 0.02,
    seed: int = 42,
    types: tuple[str, ...] = ANOMALY_TYPES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < rate < 1:
        raise ValueError("rate must be between 0 and 1")
    if not types:
        raise ValueError("types must not be empty")

    out = df.copy().reset_index(drop=True)
    n = len(out)
    k = max(4, int(round(n * rate)))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=min(k, n), replace=False))

    labels = pd.DataFrame(
        {"is_injected": [False] * n, "anomaly_type": pd.Series([""] * n, dtype="object")}
    )

    for pos, i in enumerate(idx):
        kind = types[pos % len(types)]
        amount = float(out.at[i, "amount"])
        if kind == "spike":
            out.at[i, "amount"] = amount * rng.uniform(8, 20)
        elif kind == "duplicate":
            j = max(i - 1, 0)
            out.at[i, "amount"] = float(out.at[j, "amount"])
        elif kind == "round_burst":
            out.at[i, "amount"] = max(500_000.0, round(amount / 500_000) * 500_000)
        elif kind == "drip":
            out.at[i, "amount"] = round(float(rng.uniform(1, 1000)), 2)
        else:
            raise ValueError(f"unknown type: {kind}")

        labels.at[i, "is_injected"] = True
        labels.at[i, "anomaly_type"] = kind

    return out, labels
