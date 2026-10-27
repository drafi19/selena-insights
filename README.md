# Selena Insights (work in progress)

Personal transaction anomaly detection — rebuilt from my Bangkit capstone into something
reproducible, evaluated and deployable.

**Status:** data layer + features under construction. Numbers land in `reports/`.

- Data: my own BudgetBakers Wallet account, read-only via REST API. No one else's transactions.
- Model: autoencoder (6-32-16-8-16-32-6) vs z-score and IsolationForest.
- Evaluation: injected anomalies with fixed seeds, PR-AUC, per-type recall, a per-category
  false-positive check, plus 30 hand-reviewed flags.

Full plan: `.hermes/plans/2026-10-27_091556-selena-insights-portfolio-revival.md` (local).
