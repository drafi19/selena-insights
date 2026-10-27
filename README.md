# Selena Insights (work in progress)

Personal transaction anomaly detection — rebuilt from my Bangkit capstone into something
reproducible, evaluated and deployable.

**Status:** data layer, model, evaluation and the scoring API all work.

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

## Run it

```bash
uv sync
uv run python -m selena_ml.data.wallet_client      # read-only pull, needs ~/.wallet/token.txt
uv run python -m selena_ml.data.prepare            # transfer / lending / income filtered out
uv run python -m selena_ml.train --out artifacts/v0.3.0 --version v0.3.0
uv run python -m scripts.export_onnx --artifact artifacts/v0.3.0
uv run uvicorn selena_ml.serve.app:app --port 8000
```

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/score -H 'content-type: application/json' \
  -d '{"transactions":[{"date":"2026-01-01","amount":20000,"category":"Coffee"},
                       {"date":"2026-01-02","amount":900000,"category":"Groceries"}]}'
```

`/score` returns one row per input, in input order, with `detector`, `threshold`, `score` and
`is_anomaly`. Default detector is `zscore` - that is what won on the evidence, not the autoencoder
(`docs/adr/0002-detector-choice.md`). `/` serves a dashboard on synthetic rows; `/metrics` returns
the offline evaluation numbers.

Latency, in-process benchmark (`uv run python -m scripts.bench`, 50 rows/request, 100 requests):
**p50 27 ms, p95 41 ms**. Most of it is pandas feature building, not the model.

The Docker image for the demo lives in `deploy/space/`. It was **not** built on this machine
(no Docker installed); it will be built by the Hugging Face Spaces runner.

Full plan: `.hermes/plans/2026-10-27_091556-selena-insights-portfolio-revival.md` (local).
