# Model card - Selena Insights v0.4.0

## Intended use
Flag unusual spending in one person's own transaction history, so they can review it. The model
scores each transaction against that person's recent behaviour; it is a triage aid for a human.

## Out of scope
Not a credit score, not fraud detection for a bank, not a decision system. It has never seen
another person's data and makes no claim about anyone else. Do not use it to approve, deny or
price anything.

## Data
- **Source:** my own BudgetBakers Wallet account, pulled read-only through the REST API.
- **Size:** 567 records over 92 days (2026-07-27 .. 2026-10-26). After filtering: **251 spending
  transactions**, split 175 / 38 / 38 by time.
- **Excluded before modelling:** 82 transfers between my own accounts, 105 lending-ledger records
  (`Lending, renting` + `Loan, interests` + label `Titip`), 126 income records, 3 zero-amount
  placeholders. Without these filters ~40% of "spending" is not spending.
- **Distribution:** median Rp12.000, p90 Rp63.900, max Rp650.000; 65% of rows below Rp20.000.
- **No other person's data is in this repository.** Raw and processed files are gitignored.

## Architecture
Two serving paths, one pipeline.

- **Served default: z-score over the amount channel.** The score is `max |z|` computed only over
  `amount_z_30` and `amount_z_category_30`, using training statistics stored in
  `artifacts/v0.4.0/zscore.json`. Threshold 3.0. `days_since_prev` is still computed and still shown
  to the human reviewer as `gap_days`, but it no longer enters the score - see
  `docs/adr/0003-amount-channel-and-warm-evaluation.md`.
- **Kept as the experiment: MLP autoencoder**, 7 -> 32 -> 16 -> 8 -> 16 -> 32 -> 7, ReLU, sigmoid
  output, **1.839 parameters**, same shape as the original capstone model (6 inputs, 1.774
  parameters) with one extra input for the category-relative feature. Trained with Adam (lr 1e-3,
  weight decay 1e-4), batch 32, early stopping on validation loss (patience 20). Exported to ONNX
  (opset 17); parity with PyTorch is 1.2e-7.

## Metrics
Pooled over 20 injection replicates (760 labels, 10.5% positive). **The primary table uses warm
features**: features are built once over the whole series and sliced by split, and each injected
test frame is scored together with the real history that precedes it. Appendix A0 of
`reports/v0.4/eval.md` keeps the cold numbers, where features restart at every split boundary.

| Detector | PR-AUC (warm) | ROC-AUC (warm) | PR-AUC (cold) | ROC-AUC (cold) |
|---|---|---|---|---|
| Autoencoder (this model) | 0.200 | 0.692 | 0.154 | 0.570 |
| z-score, amount channel (served) | **0.820** | **0.914** | 0.436 | 0.762 |
| IsolationForest baseline | 0.549 | 0.860 | 0.318 | 0.762 |

At the shipped threshold of 3.0, the warm z-score flags 3 of 38 test rows per replicate: pooled
recall 0.650, precision 1.000.

**The autoencoder still loses.** With ~7,000 training rows (synthetic corpus) it wins (PR-AUC 0.215
vs 0.173), so the limitation is sample size, not architecture. Serving therefore defaults to the
detector that won on evidence; the autoencoder remains available through the `detector` field.
Details: `reports/ablation.md`, `reports/score_ablation.md`, `docs/adr/0002-detector-choice.md`.

## Bias and risks

**v0.3.0 was ranking the wrong thing, and this is the record of it.** Manual review of the 30
highest-scoring rows gave precision@30 = 0.133 (4 true positives); 25 of the 30 were driven by
`days_since_prev` rather than by the amount, and the 60-day clip collapsed 24 of the top 30 onto
identical scores. v0.4.0 removes that feature from the score. The flag order changed entirely - the
30 highest warm scores are now driven only by the two amount features, with 2 ties - but the
human-labelled number for v0.4.0 is not measured yet: `reports/manual_review_v0.4.md` is generated
and awaiting verdicts. Until those verdicts exist, precision@30 for this version is unknown, not
"improved".
1. **One person, one population.** No generalisation claim is possible or intended.
2. **Synthetic labels.** Anomalies are injected (spike, duplicate, round-number burst, micro-drip),
   so the metrics measure sensitivity to patterns I defined - not to real fraud.
3. **Manual logging.** Transactions that were never recorded do not exist. The series is only as
   complete as my bookkeeping.
4. **Heavy tail.** Large purchases are normal for this user, so magnitude alone is a weak signal;
   the detector will always miss "unusual for me" anomalies that look like ordinary amounts.
5. **Small test set.** 38 rows; every per-replicate number is noisy, which is why the pooled
   protocol exists and why no single-run figure is quoted alone.
6. **Context is part of the model.** The rolling features need at least five prior rows. A stateless
   request that scores 60 rows in one batch starts cold and will flag far less than the daily
   scorer, which keeps 120 days of context. The demo page is the cold case.

## Reproduction
```bash
uv sync
uv run python -m selena_ml.data.wallet_client     # needs ~/.wallet/token.txt (read-only token)
uv run python -m selena_ml.data.prepare
uv run python -m selena_ml.train --out artifacts/v0.4.0 --version v0.4.0
uv run python -m scripts.export_onnx --artifact artifacts/v0.4.0
uv run python -m scripts.eval_protocol --artifact artifacts/v0.4.0 --out reports/v0.4
uv run python -m scripts.score_ablation --artifact artifacts/v0.4.0
```
Seeds live in `selena_ml/config.py`; the training run is deterministic given the same data file
(its SHA-256 is recorded in `artifacts/v0.4.0/metadata.json`).
