# Manual review - human-labelled precision

Reviewed **30** top flags from the shipped artefact (`artifacts/v0.3.0`, z-score detector).
A flag counts as a true positive only if the account owner judged the
transaction genuinely unusual - not merely high-scoring.

| verdict | n | share |
|---|---|---|
| y - genuinely unusual | 4 | 13.3% |
| n - normal | 26 | 86.7% |
| m - borderline | 0 | 0.0% |

**precision@30 = 0.133** (4/30).

## Which feature drove the flags

| driver | flagged | true positive | precision |
|---|---|---|---|
| `days_since_prev` | 25 | 3 | 0.12 |
| `amount_z_category_30` | 4 | 1 | 0.25 |
| `amount_z_30` | 1 | 0 | 0.00 |

## What this says

- The top of the ranking is dominated by `days_since_prev`: the detector
  mostly notices long gaps between transactions, not unusual amounts.
  A gap is not an anomaly - it is a holiday.
- Three of the four true positives sit at 3.05, barely above the 3.0
  threshold, and are also gap-driven. Only one true positive (a restaurant
  charge, 5.0 sigma against its category) was amount-driven.
- `days_since_prev` is clipped at 60 days, so long gaps collapse onto
  identical scores, which is why the ranking contains ties.

This is a finding for v0.4, not something to tune away: the feature that
carries most of the signal measures the least interesting thing.
