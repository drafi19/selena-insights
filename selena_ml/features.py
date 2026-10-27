"""Causal feature builder.

Rule: the features of row i may only use rows <= i. Output row order matches input order
(so the API response lines up 1:1 with the request), even though the maths runs in time order.

Methodological note: rolling statistics do NOT carry across splits. Each call starts cold
(`min_periods=5`), so the first rows of val/test have `amount_z_30 = 0`. That is deliberate for
offline evaluation (no information from the future leaking backwards); the effect is recorded in
`docs/MODEL_CARD.md` as a warm-up limitation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from selena_ml.config import FEATURES, GAP_CLIP_DAYS, ROLLING_WINDOW


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ("date", "amount") if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if len(df) == 0:
        return pd.DataFrame(columns=FEATURES)

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["amount"] = out["amount"].astype(float).abs()
    out["_pos"] = np.arange(len(out))

    # stable sort: transactions on the same date keep their original order
    out = out.sort_values(["date", "_pos"], kind="mergesort")

    out["log_amount"] = np.log1p(out["amount"])
    out["dow"] = out["date"].dt.dayofweek
    out["dom"] = out["date"].dt.day
    out["month"] = out["date"].dt.month

    gap = out["date"].diff().dt.days
    out["days_since_prev"] = gap.clip(lower=0, upper=GAP_CLIP_DAYS).fillna(0.0)

    # statistics from PREVIOUS transactions only (shift before rolling)
    prev = out["log_amount"].shift(1)
    mean = prev.rolling(ROLLING_WINDOW, min_periods=5).mean()
    std = prev.rolling(ROLLING_WINDOW, min_periods=5).std(ddof=0)
    z = (out["log_amount"] - mean) / std.replace(0.0, np.nan)
    out["amount_z_30"] = z.fillna(0.0).clip(-5.0, 5.0)

    feats = out.loc[:, ["_pos", *FEATURES]].sort_values("_pos")
    return feats.loc[:, FEATURES].reset_index(drop=True)
