# Data

Source: **my own BudgetBakers Wallet account**, pulled read-only through the REST API
(`selena_ml/data/wallet_client.py`). No other person's transactions are in this repository.

| Artefact | Content | Committed? |
|---|---|---|
| `data/raw/wallet_raw.json` | raw API response | **no** (gitignored) |
| `data/processed/spending.parquet` | spending series after filtering | **no** (gitignored) |
| `docs/WALLET_AUDIT.md` | aggregates: counts, range, categories, duplicates | yes |
| `web/sample.json` | 60 **synthetic** transactions for the public demo | yes |

## The four filters that decide what "spending" means
Measured on 567 records (2026-10-27):

1. `transfer` is set -> dropped. Moving money between my own accounts is not spending (**82 records**).
2. category `Lending, renting` or `Loan, interests`, or label `Titip` -> dropped. That is the
   lending ledger, not consumption (**105 records**).
3. `record_type != "expense"` -> dropped, the income series is out of scope for v0.1 (**126**).
4. `amount == 0` -> dropped, placeholder rows (**3**).

Result: **251 spending rows**. Without filters 1 and 2, roughly 40% of what the model sees would
not be spending at all - which is exactly the mistake the original capstone pipeline made.

## Limits to quote whenever numbers are published
1. **One person.** The model scores a user against their own history; no cross-user claim.
2. **92 days only** (2026-07-27 .. 2026-10-26). Not enough for seasonality claims; retrain as it grows.
3. **Manual logging.** Unrecorded transactions do not exist.
4. **No reliable time of day.** `createdAt` is when I typed it, not when I paid, so features use
   dates only (median backfill gap 1.25 days, 88 records entered >7 days late).
5. **The public repo contains aggregates and synthetic rows only**, never a real record.
