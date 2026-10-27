"""Read-only client for the BudgetBakers Wallet REST API.

GET only. There is no POST/PATCH/DELETE here - the write path lives in `~/.wallet/`.
The token is read from a file, never from a committed env var.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import requests

BASE = "https://rest.budgetbakers.com/wallet"
DEFAULT_TOKEN = Path.home() / ".wallet" / "token.txt"
TZ = "Asia/Jakarta"


def load_token(path: str | Path | None = None) -> str:
    p = Path(path) if path else Path(os.environ.get("WALLET_TOKEN_FILE", DEFAULT_TOKEN))
    if not p.exists():
        raise FileNotFoundError(f"token not found: {p}")
    token = p.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"empty token: {p}")
    return token


def fetch_records(token: str, page_size: int = 100, max_pages: int = 200) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    out: list[dict] = []
    offset = 0
    for _ in range(max_pages):
        resp = requests.get(
            f"{BASE}/v1/api/records",
            params={"limit": page_size, "offset": offset},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        batch = body.get("records") or []
        out.extend(batch)
        nxt = body.get("nextOffset")
        if not batch or nxt is None:
            break
        offset = int(nxt)
    return out


def _local(ts: str) -> pd.Timestamp:
    return pd.to_datetime(ts, utc=True).tz_convert(TZ).tz_localize(None)


def to_frame(records: list[dict]) -> pd.DataFrame:
    """Normalise the API payload. No filtering decisions in here."""
    rows = []
    for r in records:
        cat = r.get("category") or {}
        labels = r.get("labels") or []
        rows.append(
            {
                "id": r.get("id"),
                "date": _local(r.get("recordDate")),
                "created_at": _local(r.get("createdAt")),
                "amount": float((r.get("amount") or {}).get("value", 0.0)),
                "currency": (r.get("amount") or {}).get("currencyCode"),
                "record_type": r.get("recordType"),
                "account": r.get("accountName"),
                "category": cat.get("name"),
                "counterparty": r.get("counterParty"),
                "note": r.get("note"),
                "labels": ",".join(
                    str(lab.get("name", "")) for lab in labels if isinstance(lab, dict)
                ),
                "is_transfer": bool(r.get("transfer")),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("no records to normalise")
    return df.sort_values(["date", "id"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/wallet_raw.json")
    ap.add_argument("--token-file", default=None)
    args = ap.parse_args()

    token = load_token(args.token_file)
    records = fetch_records(token)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    df = to_frame(records)
    print(f"{len(df)} records -> {out}")
    lo, hi = df["date"].min().date(), df["date"].max().date()
    print(f"{lo} .. {hi} | {int(df['is_transfer'].sum())} transfers")


if __name__ == "__main__":
    main()
