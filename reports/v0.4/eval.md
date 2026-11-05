# Evaluation - v0.4.0 - warm features

Model trained on real Wallet spending (n=175). Test n=38; injection 10% x 20 replicates (~4 labels per replicate).

Features are **warm**: tables A and A2 build features once over the full series and slice
by split position, and every injection replicate uses the real history that precedes the
test window. The cold numbers in appendix A0 rebuild features separately per split, which
restarts the rolling windows at zero. That cold start is an artefact of rebuilding
features per split, not a property of the detector: the served daily scorer keeps context
(`scripts/daily_score.py --context-days`), while the demo page does not. That difference
is why the demo shows few flags.

## A. Real test set (warm features)

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | P@10 | Flagged |
|---|---|---|---|---|---|---|---|
| autoencoder | 0.267 +- 0.125 | 0.689 +- 0.114 | 0.100 +- 0.300 | 0.025 +- 0.075 | 0.040 +- 0.120 | 0.170 +- 0.078 | 1 |
| zscore | 0.825 +- 0.123 | 0.914 +- 0.094 | 1.000 +- 0.000 | 0.650 +- 0.200 | 0.769 +- 0.161 | 0.350 +- 0.059 | 3 |
| isolation_forest | 0.587 +- 0.116 | 0.860 +- 0.098 | 0.408 +- 0.153 | 0.300 +- 0.127 | 0.343 +- 0.134 | 0.330 +- 0.064 | 3 |

## A2. Pooled across 20 replicates (~760 labels)

One number per model, computed from scores and labels concatenated across replicates.
This is the number quoted in the README; table A shows the spread.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| autoencoder | 0.200 | 0.692 | 0.105 | 0.025 | 0.040 |
| zscore | 0.820 | 0.914 | 1.000 | 0.650 | 0.788 |
| isolation_forest | 0.549 | 0.860 | 0.414 | 0.300 | 0.348 |

## A0. Cold-start baseline (old path, not primary)

The same detectors and seeds, but features are rebuilt separately on train, val, test and
every injected test frame, so the rolling windows start cold. This is the old number set;
compare it with the warm numbers in A2.

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| autoencoder | 0.154 | 0.570 |
| zscore | 0.436 | 0.762 |
| isolation_forest | 0.318 | 0.762 |

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
| autoencoder | 0.195 +- 0.033 | 0.445 +- 0.131 | 0.107 +- 0.054 | 0.171 +- 0.081 |
| zscore | 0.184 +- 0.021 | 0.163 +- 0.079 | 0.027 +- 0.011 | 0.046 +- 0.019 |
| isolation_forest | 0.210 +- 0.022 | 0.271 +- 0.068 | 0.040 +- 0.011 | 0.069 +- 0.018 |

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
