# Ablation: does the category-relative feature help?

Both models are trained on the same rows and scored with the same protocol (20 injection
replicates, pooled PR-AUC over ~760 labels). Only the feature set differs.

## Pooled PR-AUC (real Wallet data, test n=38)

| Feature set | Autoencoder | z-score | IsolationForest |
|---|---|---|---|
| v0.1.0 (6 features) | 0.170 | 0.296 | 0.320 |
| v0.2.0 (7 features, + category z) | 0.154 | 0.296 | 0.318 |

## Pooled ROC-AUC

| Feature set | Autoencoder | z-score | IsolationForest |
|---|---|---|---|
| v0.1.0 (6 features) | 0.598 | 0.790 | 0.775 |
| v0.2.0 (7 features, + category z) | 0.570 | 0.791 | 0.762 |

## Synthetic corpus (10k rows, separate model per version)

| Feature set | Autoencoder PR-AUC |
|---|---|
| v0.1.0 (6 features) | 0.197 |
| v0.2.0 (7 features, + category z) | 0.215 |

## Reading
- A feature counts only if the pooled number goes up; here it did not.
- Spread is in `reports/eval.json`; a single pooled number without it is not enough.
- The synthetic table is the control: with ~7,000 training rows the autoencoder beats the
  baselines, so the gap is a sample-size problem, not an architecture problem.
- Caveat: the simulator has no category column, so the 7th feature is degenerate there.
  The 0.197 -> 0.215 difference is run-to-run variance, not evidence about the feature.
- The category feature is active on 56.6% of real rows (13 of 31 categories have >= 6
  transactions), so a zero effect is not a dead column: not enough data per category.
- Decision recorded in `docs/adr/0002-detector-choice.md`.
