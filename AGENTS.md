# AGENTS.md — rules for this repo

## Canonical commands
- Install: `uv sync`
- Test: `uv run pytest -q` (must be green before any commit)
- Lint: `uv run ruff check .`
- Pull data: `uv run python -m selena_ml.data.wallet_client`
- Prepare: `uv run python -m selena_ml.data.prepare`
- Audit: `uv run python -m scripts.audit_wallet`

## Rules
1. TDD: write the test, watch it fail, implement, watch it pass, commit.
2. Never write a metric that did not come out of `reports/eval.json`.
3. Never commit `data/raw/`, `data/processed/`, `.env`, or any credential.
4. Features live only in `selena_ml/config.py`. Adding one means config + test + model card.
5. Every random number goes through `SEED` in `config.py`.
6. If a test fails because of the data, fix the data - never weaken the assertion.
7. Never print individual amounts, counterparties or notes from Wallet data into logs,
   reports, commits or answers. Aggregates only.
