"""Score the last N days and write the top flags to a CSV for manual review.

This produces the only human-labelled number in the project. Run:

    uv run python -m scripts.review_flags --top 30
Then fill the `verdict` column (y / n / m) and compute (verdict == "y").mean().
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from selena_ml.data.load import load_transactions
from selena_ml.features import build_features
from selena_ml.scaling import MinMaxJSON


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/v0.3.0")
    ap.add_argument("--data", default="data/processed/spending.parquet")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--detector", default="zscore")
    ap.add_argument("--out", default="reports/manual_review.csv")
    args = ap.parse_args()

    art = Path(args.artifact)
    scaler = MinMaxJSON.from_json(art / "scaler.json")

    if args.detector != "zscore":
        raise SystemExit("only the z-score detector is supported here (it is the default)")

    stats = json.loads((art / "zscore.json").read_text())
    mu = np.asarray(stats["mu"], dtype=float)
    sd = np.asarray(stats["sd"], dtype=float)
    threshold = float(stats["threshold"])

    df = load_transactions(args.data)
    cutoff = df["date"].max() - pd.Timedelta(days=args.days)
    recent = df[df["date"] >= cutoff].reset_index(drop=True)

    S = scaler.transform(build_features(recent).to_numpy())
    scores = np.abs((S - mu) / sd).max(axis=1)

    out = recent.assign(score=scores, flagged=scores > threshold)
    out = out.sort_values("score", ascending=False).head(args.top)
    cols = [c for c in ("date", "amount", "category", "score", "flagged") if c in out.columns]
    out = out[cols].assign(verdict="")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"{len(out)} top flags -> {path}")
    print(f"flagged among top-{args.top}: {int(out['flagged'].sum())}")
    print("Fill `verdict` with y / n / m, then precision@k = (verdict == 'y').mean()")


if __name__ == "__main__":
    main()
