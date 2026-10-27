"""Turn a Wallet dump into a spending series that is actually spending.

Four filters that must not be skipped:
1. transfers between own accounts -> not spending
2. `Lending, renting` / `Loan, interests` / label `Titip` -> lending ledger, not consumption
3. record_type other than 'expense' -> income series lives elsewhere (out of scope for v0.1)
4. amount 0 -> placeholder records
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from selena_ml.data.wallet_client import to_frame

PASS_THROUGH_CATEGORIES = {"Lending, renting", "Loan, interests"}
PASS_THROUGH_LABELS = {"Titip"}


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (spending, excluded). `excluded` is for reporting, never for training."""
    work = df.copy()
    reason = pd.Series([""] * len(work), index=work.index, dtype="object")

    transfer = work["is_transfer"].fillna(False).astype(bool)
    reason[transfer] = "transfer"

    pass_cat = work["category"].isin(PASS_THROUGH_CATEGORIES) & ~transfer
    reason[pass_cat] = "pass_through_category"

    pattern = "|".join(sorted(PASS_THROUGH_LABELS))
    pass_label = work["labels"].str.contains(pattern, na=False) & (reason == "")
    reason[pass_label] = "pass_through_label"

    income = (work["record_type"] != "expense") & (reason == "")
    reason[income] = "not_expense"

    zero = (work["amount"].abs() <= 0) & (reason == "")
    reason[zero] = "zero_amount"

    keep = reason == ""
    spending = pd.DataFrame(
        {
            "date": work.loc[keep, "date"],
            "amount": work.loc[keep, "amount"].abs(),
            "category": work.loc[keep, "category"].fillna("unknown"),
        }
    ).sort_values("date", kind="mergesort").reset_index(drop=True)

    excluded = work.loc[~keep].assign(reason=reason[~keep])

    if spending.empty:
        raise ValueError("no spending records left after filtering - check the dump")
    if (spending["amount"] <= 0).any():
        raise ValueError("amount <= 0 slipped through the filter")
    return spending, excluded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="data/raw/wallet_raw.json")
    ap.add_argument("--out", default="data/processed/spending.parquet")
    args = ap.parse_args()

    df = to_frame(json.loads(Path(args.src).read_text(encoding="utf-8")))
    spending, excluded = prepare(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    spending.to_parquet(out, index=False)
    print(f"spending={len(spending)} -> {out}")
    print(f"excluded={len(excluded)}: {excluded['reason'].value_counts().to_dict()}")
    print(f"range={spending['date'].min().date()}..{spending['date'].max().date()}")


if __name__ == "__main__":
    main()
