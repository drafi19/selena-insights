"""Single loader for the spending series. Every data entry point goes through this."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = ("date", "amount")


def load_transactions(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"data not found: {p}")
    df = pd.read_parquet(p)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{p.name}: missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["amount"] = df["amount"].astype(float)
    if (df["amount"] <= 0).any():
        raise ValueError(f"{p.name}: non-positive amount - prepare() should have filtered it")

    return df.sort_values("date", kind="mergesort").reset_index(drop=True)
