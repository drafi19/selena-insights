"""Audit the Wallet dump: counts, range, composition, quality. Never writes amounts or names.

Run: uv run python scripts/audit_wallet.py > docs/WALLET_AUDIT.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from selena_ml.data.prepare import prepare
from selena_ml.data.wallet_client import to_frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="data/raw/wallet_raw.json")
    args = ap.parse_args()

    df = to_frame(json.loads(Path(args.src).read_text(encoding="utf-8")))
    days = (df["date"].max() - df["date"].min()).days + 1
    gap = (df["created_at"] - df["date"]).dt.total_seconds() / 86400

    print("# Wallet data audit\n")
    print(f"- records: **{len(df)}**")
    print(f"- range: {df['date'].min().date()} .. {df['date'].max().date()} ({days} days)")
    print(f"- rate: {len(df) / days:.1f} records/day")
    print(f"- transfers: {int(df['is_transfer'].sum())} ({df['is_transfer'].mean():.1%})")
    print(f"- record types: {df['record_type'].value_counts().to_dict()}")
    print(f"- distinct categories: {df['category'].nunique()}")
    print(f"- labelled: {(df['labels'] != '').mean():.1%}")
    print(f"- backfill > 7 days: {int((gap > 7).sum())} records (median {gap.median():.2f} days)")
    print(f"- accounts: {df['account'].nunique()}")

    print("\n## Duplicate candidates (same account + amount + day)\n")
    day = df["date"].dt.date
    dup = df.assign(day=day).duplicated(subset=["account", "amount", "day"], keep=False)
    print(f"- rows involved: {int(dup.sum())} of {len(df)}")
    spending, excluded = prepare(df)
    print("\n## After filtering\n")
    print(f"- spending records: **{len(spending)}**")
    print(f"- excluded: {excluded['reason'].value_counts().to_dict()}")
    print(f"- spending range: {spending['date'].min().date()} .. {spending['date'].max().date()}")

    print("\n> No individual amounts, counterparties or notes are written to this report.")


if __name__ == "__main__":
    main()
