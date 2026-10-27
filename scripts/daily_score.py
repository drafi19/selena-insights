"""Score the last N days from Wallet and print a short summary. Used by the nightly cron.

Read-only. The output is meant for a private chat - never write it into the repo.
Run: uv run python -m scripts.daily_score --top 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from selena_ml.data.prepare import prepare
from selena_ml.data.wallet_client import fetch_records, load_token, to_frame
from selena_ml.features import build_features
from selena_ml.scaling import MinMaxJSON


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/v0.3.0")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--context-days", type=int, default=120)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    art = Path(args.artifact)
    scaler = MinMaxJSON.from_json(art / "scaler.json")
    stats = json.loads((art / "zscore.json").read_text())
    mu = np.asarray(stats["mu"], dtype=float)
    sd = np.asarray(stats["sd"], dtype=float)
    threshold = float(stats["threshold"])

    df, _ = prepare(to_frame(fetch_records(load_token())))
    if df.empty:
        raise ValueError("no spending records - check the token or the filters")

    ctx_start = df["date"].max() - pd.Timedelta(days=args.context_days)
    ctx = df[df["date"] >= ctx_start].reset_index(drop=True)

    S = scaler.transform(build_features(ctx).to_numpy())
    scores = np.abs((S - mu) / sd).max(axis=1)

    scored = ctx.assign(score=scores, flagged=scores > threshold)
    recent_start = df["date"].max() - pd.Timedelta(days=args.days)
    recent = scored[scored["date"] >= recent_start]
    flags = recent[recent["flagged"]].sort_values("score", ascending=False)

    print(f"transactions {args.days}d: {len(recent)}")
    print(f"flags: {len(flags)}")
    if flags.empty:
        print("no anomalies")
        return
    for row in flags.head(args.top).itertuples():
        when = row.date.date()
        print(f"- {when} | {row.category} | Rp{int(row.amount):,} | score {row.score:.3f}")


if __name__ == "__main__":
    main()
