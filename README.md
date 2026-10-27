# Selena Insights (work in progress)

Personal transaction anomaly detection — rebuilt from my Bangkit capstone into something
reproducible, evaluated and deployable.

**Status:** data layer, model and evaluation are done. Serving is next.

- Data: my own BudgetBakers Wallet account, read-only via REST API. No one else's transactions.
- Model: autoencoder (6-32-16-8-16-32-6) vs z-score and IsolationForest.
- Evaluation: injected anomalies with fixed seeds, PR-AUC, per-type recall, a per-category
  false-positive check, plus 30 hand-reviewed flags.

## Results (251 real transactions, 20 injection replicates, pooled PR-AUC)

| Detector | PR-AUC | ROC-AUC | Recall @ FPR 1% |
|---|---|---|---|
| Autoencoder (6 features) | 0.170 | 0.598 | 0.025 |
| Autoencoder (7 features) | 0.154 | 0.570 | 0.025 |
| z-score | 0.296 | 0.791 | 0.138 |
| IsolationForest | 0.320 | 0.762 | 0.037 |

Chance level is 0.105 (10.5% injected positives). **The autoencoder loses.** On a 10,000-row
synthetic corpus with ~7,000 training rows it wins (PR-AUC 0.197 vs 0.173), so the gap is a
sample-size problem: 175 training rows cannot support a 1,839-parameter model.

Adding a category-relative feature did not help and is published as a negative ablation:
`reports/ablation.md`. Decision: `docs/adr/0002-detector-choice.md`.

Full numbers: `reports/eval.md`, `reports/v0.2/eval.md`, `reports/eval.json`.

Full plan: `.hermes/plans/2026-10-27_091556-selena-insights-portfolio-revival.md` (local).
