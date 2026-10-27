# Model card - Selena Insights v0.3.0

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
MLP autoencoder, 7 -> 32 -> 16 -> 8 -> 16 -> 32 -> 7, ReLU, sigmoid output, **1.839 parameters**.
Same shape as the original capstone model (6 inputs, 1.774 parameters), one extra input for the
category-relative feature. Trained with Adam (lr 1e-3, weight decay 1e-4), batch 32, early
stopping on validation loss (patience 20). 123 epochs in the shipped run; best validation loss
0.0367. Exported to ONNX (opset 17) for serving; parity with PyTorch is 1.2e-7.

## Metrics
Pooled over 20 injection replicates (760 labels, 10.5% positive), evaluated with
`scripts.eval_protocol`:

| Detector | PR-AUC | ROC-AUC | Recall @ FPR 1% |
|---|---|---|---|
| Autoencoder (this model) | 0.154 | 0.570 | 0.025 |
| z-score baseline | 0.296 | 0.791 | 0.138 |
| IsolationForest baseline | 0.318 | 0.762 | 0.037 |

**The autoencoder loses to a z-score on this data.** With ~7,000 training rows (synthetic corpus)
it wins (PR-AUC 0.215 vs 0.173), so the limitation is sample size, not architecture. Serving
therefore defaults to the z-score detector; the autoencoder remains available for comparison.
Details: `reports/ablation.md`, `docs/adr/0002-detector-choice.md`.

## Bias and risks
1. **One person, one population.** No generalisation claim is possible or intended.
2. **Synthetic labels.** Anomalies are injected (spike, duplicate, round-number burst, micro-drip),
   so the metrics measure sensitivity to patterns I defined - not to real fraud.
3. **Manual logging.** Transactions that were never recorded do not exist. The series is only as
   complete as my bookkeeping.
4. **Heavy tail.** Large purchases are normal for this user, so magnitude alone is a weak signal;
   the detector will always miss "unusual for me" anomalies that look like ordinary amounts.
5. **Small test set.** 38 rows; every per-replicate number is noisy, which is why the pooled
   protocol exists and why no single-run figure is quoted alone.

## Reproduction
```bash
uv sync
uv run python -m selena_ml.data.wallet_client     # needs ~/.wallet/token.txt (read-only token)
uv run python -m selena_ml.data.prepare
uv run python -m selena_ml.train --out artifacts/v0.3.0 --version v0.3.0
uv run python -m scripts.export_onnx --artifact artifacts/v0.3.0
uv run python -m scripts.eval_protocol --artifact artifacts/v0.3.0
```
Seeds live in `selena_ml/config.py`; the training run is deterministic given the same data file
(its SHA-256 is recorded in `artifacts/v0.3.0/metadata.json`).
