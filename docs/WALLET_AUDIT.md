# Wallet data audit

- records: **567**
- range: 2026-07-27 .. 2026-10-26 (92 days)
- rate: 6.2 records/day
- transfers: 82 (14.5%)
- record types: {'expense': 348, 'income': 219}
- distinct categories: 37
- labelled: 33.0%
- backfill > 7 days: 88 records (median 1.25 days)
- accounts: 6

## Duplicate candidates (same account + amount + day)

- rows involved: 124 of 567

## After filtering

- spending records: **251**
- excluded: {'not_expense': 126, 'pass_through_category': 105, 'transfer': 82, 'zero_amount': 3}
- spending range: 2026-07-27 .. 2026-10-26

> No individual amounts, counterparties or notes are written to this report.
