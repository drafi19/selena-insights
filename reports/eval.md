# Evaluation - v0.1.0

Model trained on real Wallet spending (n=175). Test n=38; injection 10% x 20 replicates (~4 labels per replicate).

## A. Real test set

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | P@10 | Flagged |
|---|---|---|---|---|---|---|---|
| autoencoder | 0.233 +- 0.114 | 0.593 +- 0.117 | 0.100 +- 0.300 | 0.025 +- 0.075 | 0.040 +- 0.120 | 0.140 +- 0.073 | 0 |
| zscore | 0.364 +- 0.107 | 0.789 +- 0.079 | 0.167 +- 0.167 | 0.125 +- 0.125 | 0.143 +- 0.143 | 0.305 +- 0.059 | 3 |
| isolation_forest | 0.380 +- 0.115 | 0.772 +- 0.092 | 0.150 +- 0.357 | 0.037 +- 0.089 | 0.060 +- 0.143 | 0.300 +- 0.063 | 0 |

## A2. Pooled across 20 replicates (~760 labels)

One number per model, computed from scores and labels concatenated across replicates.
This is the number quoted in the README; table A shows the spread.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| autoencoder | 0.170 | 0.598 | 0.222 | 0.025 | 0.045 |
| zscore | 0.296 | 0.790 | 0.167 | 0.125 | 0.143 |
| isolation_forest | 0.320 | 0.775 | 1.000 | 0.037 | 0.072 |

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
| autoencoder | 0.197 +- 0.023 | 0.420 +- 0.074 | 0.138 +- 0.041 | 0.207 +- 0.056 |
| zscore | 0.173 +- 0.015 | 0.098 +- 0.055 | 0.031 +- 0.017 | 0.047 +- 0.026 |
| isolation_forest | 0.189 +- 0.022 | 0.211 +- 0.063 | 0.036 +- 0.017 | 0.060 +- 0.027 |

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
