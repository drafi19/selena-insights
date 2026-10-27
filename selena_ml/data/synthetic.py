"""Deterministic synthetic transactions: a test fixture generator and a corpus simulator.

Neither is used to train the shipped model. `make_synthetic` is for tests; `simulate_transactions`
feeds the public demo, cold-start examples and scale tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(n: int = 240, seed: int = 7) -> pd.DataFrame:
    """Small deterministic frame for tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    weekday = dates.dayofweek.to_numpy()
    base = rng.lognormal(mean=12.0, sigma=0.6, size=n)
    base = base * np.where(weekday >= 5, 1.4, 1.0)
    amount = np.round(base / 1000) * 1000
    direction = np.where(dates.day.to_numpy() == 25, "income", "expense")
    return pd.DataFrame({"date": dates, "amount": amount, "direction": direction})


WEEKEND_MULT = 0.8

EXPENSE_REGIMES = (
    (0.62, 8.9, 0.8),    # small: parking, coffee, transport
    (0.25, 9.9, 0.7),    # daily groceries
    (0.09, 11.2, 0.6),   # large: electronics, rent
    (0.04, 12.4, 0.5),   # very large: transfers, down payments
)


def _draw_expense(rng: np.random.Generator, n: int) -> np.ndarray:
    upper = np.cumsum([w for w, _, _ in EXPENSE_REGIMES])
    pick = rng.random(n)
    out = np.empty(n, dtype=float)
    for (w, mu, sigma), hi in zip(EXPENSE_REGIMES, upper, strict=True):
        mask = (pick > hi - w) & (pick <= hi)
        out[mask] = rng.lognormal(mu, sigma, size=int(mask.sum()))
    return out


def simulate_transactions(
    n: int = 10_000,
    seed: int = 42,
    start: str = "2023-01-01",
    income_share: float = 0.06,
) -> pd.DataFrame:
    """Structured stress test, not a distribution clone.

    Regime weights follow the 2026-10-27 Wallet audit (median 12k, p75 27k, p90 72k).
    Used for the public demo, cold start and scale tests - never for training the shipped model.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="8h")
    is_weekend = dates.dayofweek.to_numpy() >= 5

    amount = _draw_expense(rng, n)
    # the audit shows weekend spending is LOWER for this user (0.77x), not higher
    amount = np.where(is_weekend, amount * WEEKEND_MULT, amount)

    direction = np.where(rng.random(n) < income_share, "income", "expense")
    salary = rng.lognormal(14.2, 0.45, size=n)
    amount = np.where(direction == "income", salary, amount)
    amount = np.round(amount / 1000) * 1000

    df = pd.DataFrame({"date": dates, "amount": amount, "direction": direction})
    return df.sort_values("date", kind="mergesort").reset_index(drop=True)
