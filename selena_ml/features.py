"""Causal feature builder.

Rule: the features of row i may only use rows <= i. Output row order matches input order
(so the API response lines up 1:1 with the request), even though the maths runs in time order.

Rolling statistics need history. Calling `build_features` separately on each split restarts the
windows cold (`min_periods=5`), so the first rows of val/test get `amount_z_30 = 0` even though
real transactions precede them. Use `build_features_with_history` whenever earlier rows exist.
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
    if "category" in out.columns:
        out["category"] = out["category"].fillna("unknown")
    else:
        out["category"] = "unknown"
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

    # same idea, but relative to the last transactions of the SAME category:
    # 500k for "Electronics" is normal, 500k for "Coffee" is not.
    grouped = out.groupby("category", sort=False)["log_amount"]
    mean_c = grouped.transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=5).mean()
    )
    std_c = grouped.transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=5).std(ddof=0)
    )
    z_c = (out["log_amount"] - mean_c) / std_c.replace(0.0, np.nan)
    out["amount_z_category_30"] = z_c.fillna(0.0).clip(-5.0, 5.0)

    feats = out.loc[:, ["_pos", *FEATURES]].sort_values("_pos")
    return feats.loc[:, FEATURES].reset_index(drop=True)


def build_features_with_history(history: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Build features for `rows` using `history` as context.

    Rolling windows (and the gap to the previous transaction) need earlier transactions to be
    correct. Calling `build_features` on `rows` alone starts every window cold, so a spike in
    the first few rows looks ordinary (`amount_z_30 = 0`). Concatenating the real, chronologically
    earlier `history` and building once gives each row in `rows` the same feature value it would
    have had in a single pass over the full series. The result contains only the `rows` rows, in
    their original order, with a fresh index.
    """
    if len(rows) == 0:
        return pd.DataFrame(columns=FEATURES)
    combined = pd.concat([history, rows], ignore_index=True)
    feats = build_features(combined)
    return feats.iloc[len(history) :].reset_index(drop=True)
