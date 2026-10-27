"""Score the last N days and write the top flags to a CSV for manual review.

This produces the only human-labelled number in the project. Run:

    uv run python -m scripts.review_flags --top 30
Then fill the `verdict` column (y / n / m) and compute (verdict == "y").mean().

Columns are ordered so the review is fast: `driver` says which feature pushed the score, `why`
says how far, so you can decide without re-deriving the features by hand.
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

LABELS = {
    "log_amount": "nominal absolut",
    "dow": "hari dalam minggu",
    "dom": "tanggal",
    "month": "bulan",
    "days_since_prev": "jarak dari transaksi sebelumnya",
    "amount_z_30": "nominal vs 30 transaksi terakhir",
    "amount_z_category_30": "nominal vs kategori ini",
}


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
    meta_path = art / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    features = meta.get("features") or list(LABELS)

    df = load_transactions(args.data)
    cutoff = df["date"].max() - pd.Timedelta(days=args.days)
    recent = df[df["date"] >= cutoff].reset_index(drop=True)

    S = scaler.transform(build_features(recent).to_numpy())
    z = np.abs((S - mu) / sd)
    scores = z.max(axis=1)

    driver_idx = z.argmax(axis=1)
    drivers: list[str] = []
    why: list[str] = []
    for row, col in enumerate(driver_idx):
        name = features[col]
        drivers.append(name)
        why.append(f"{LABELS.get(name, name)} {z[row, col]:.1f} sigma")

    out = recent.assign(score=scores, flagged=scores > threshold, driver=drivers, why=why)
    out = out.sort_values("score", ascending=False).head(args.top)
    keep = ("date", "amount", "category", "score", "flagged", "driver", "why")
    out = out[[c for c in keep if c in out.columns]].assign(verdict="")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    md = Path(args.out).with_suffix(".md")
    lines = [f"# Manual review - {len(out)} top flags", "",
             f"threshold {threshold:.3f} · {int(out['flagged'].sum())} above it", "",
             "| # | date | amount | category | score | driver | why | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for i, row in enumerate(out.itertuples(), 1):
        lines.append(
            f"| {i} | {row.date.date()} | Rp{int(row.amount):,} | {row.category} | "
            f"{row.score:.3f} | {row.driver} | {row.why} |  |"
        )
    lines += ["", "verdict: `y` = genuinely unusual, `n` = normal, `m` = borderline.",
              "precision@30 = (verdict == 'y').mean()"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{len(out)} top flags -> {path} (and {md.name})")
    print(f"above threshold: {int(out['flagged'].sum())}/{len(out)}")
    print("driver counts:", out["driver"].value_counts().to_dict())


if __name__ == "__main__":
    main()
