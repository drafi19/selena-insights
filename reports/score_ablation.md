# Score ablation - z-score feature sets (warm features)

Artefact `artifacts/v0.4.0`. The same real-data split and the same 20 injection seeds for every scheme. Warm features use the real history before the test window; cold rebuilds features separately per split, as the old script did. PR-AUC and ROC-AUC are pooled across replicates; ties are counted among the 30 highest warm scores on the clean 90-day window at threshold 3.0.

| Scheme | Features | PR-AUC warm | ROC-AUC warm | PR-AUC cold | ROC-AUC cold | Top-30 ties |
|---|---|---|---|---|---|---|
| full7 | log_amount, dow, dom, month, days_since_prev, amount_z_30, amount_z_category_30 | 0.511 | 0.885 | 0.296 | 0.791 | 24 |
| amount2 | amount_z_30, amount_z_category_30 | 0.820 | 0.914 | 0.436 | 0.762 | 2 |
| amount3 | log_amount, amount_z_30, amount_z_category_30 | 0.834 | 0.929 | 0.510 | 0.858 | 6 |
| no_gap | log_amount, dow, dom, month, amount_z_30, amount_z_category_30 | 0.830 | 0.919 | 0.491 | 0.825 | 6 |

## Reading

- `amount3` has the highest warm pooled PR-AUC (0.834).
- The shipped amount-only score (`amount2`) scores 0.820 warm against 0.511 for the old 7-feature score; cold those numbers are 0.436 and 0.296.
- `no_gap` removes `days_since_prev` from the full set; `amount3` adds the raw `log_amount` back to the shipped pair.
- Top-30 ties on the clean window: `full7` 24, `amount2` 2, `amount3` 6, `no_gap` 6.
- Ties matter because a long gap clipped at 60 days collapses many rows onto the same score; they are counted on the clean series only, without injected rows.
