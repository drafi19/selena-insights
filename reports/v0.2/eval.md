# Evaluation - v0.1.0

Model trained on real Wallet spending (n=175). Test n=38; injection 10% x 20 replicates (~4 labels per replicate).

## A. Real test set

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | P@10 | Flagged |
|---|---|---|---|---|---|---|---|
| autoencoder | 0.215 +- 0.125 | 0.568 +- 0.111 | 0.100 +- 0.300 | 0.025 +- 0.075 | 0.040 +- 0.120 | 0.115 +- 0.073 | 0 |
| zscore | 0.366 +- 0.109 | 0.790 +- 0.080 | 0.175 +- 0.179 | 0.138 +- 0.147 | 0.154 +- 0.160 | 0.305 +- 0.059 | 3 |
| isolation_forest | 0.382 +- 0.136 | 0.759 +- 0.104 | 0.100 +- 0.300 | 0.025 +- 0.075 | 0.040 +- 0.120 | 0.290 +- 0.054 | 0 |

## A2. Pooled across 20 replicates (~760 labels)

One number per model, computed from scores and labels concatenated across replicates.
This is the number quoted in the README; table A shows the spread.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| autoencoder | 0.154 | 0.570 | 1.000 | 0.025 | 0.049 |
| zscore | 0.296 | 0.791 | 0.180 | 0.138 | 0.156 |
| isolation_forest | 0.318 | 0.762 | 1.000 | 0.025 | 0.049 |

## B. Recall per anomaly type (autoencoder)

| Type | Recall |
|---|---|
| spike | 0.050 +- 0.218 |
| duplicate | 0.000 +- 0.000 |
| round_burst | 0.050 +- 0.218 |
| drip | 0.000 +- 0.000 |

## C. Synthetic corpus (10,000 rows, 5% injection)

A separate model is trained on this corpus; the real-data model is not transferred here.

| Model | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| autoencoder | 0.215 +- 0.024 | 0.454 +- 0.105 | 0.116 +- 0.035 | 0.183 +- 0.051 |
| zscore | 0.173 +- 0.015 | 0.098 +- 0.055 | 0.031 +- 0.017 | 0.047 +- 0.026 |
| isolation_forest | 0.227 +- 0.025 | 0.320 +- 0.128 | 0.062 +- 0.033 | 0.103 +- 0.054 |

## D. Flag rate per category (clean full series)

| Category | n | Flag rate |
|---|---|---|
| Restaurants & fast food | 46 | 2.2% |
| Groceries | 31 | 0.0% |
| Education & development | 21 | 0.0% |
| Snacks | 16 | 0.0% |
| Charity, gifts | 15 | 0.0% |
| Active sport, fitness | 14 | 0.0% |
| Non-coffee | 13 | 0.0% |
| Charges, fees | 12 | 0.0% |

Overall flag rate on the clean full series: **0.8%** (target FPR 1%).
A category far above the average means the model punishes that category, not a row.

## Limitations
- Labels come from synthetic injection, not human judgement. These numbers measure
  sensitivity to the patterns we defined, not every kind of fraud.
- Table A reuses one trained model across replicates; the variance comes from
  which rows were mutated, not from the weights (that is the seed sweep).
- Table C must never be averaged together with table A.
- Human-labelled precision lives in `reports/manual_review.md` (T25b).
