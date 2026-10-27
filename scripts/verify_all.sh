#!/usr/bin/env bash
# Reproduce every check in this repo. Two stages: hermetic (always) and full (needs private data).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/4 lint"
uv run ruff check .

echo "== 2/4 tests"
uv run pytest -q

echo "== 3/4 serving smoke (no private data needed)"
uv run python -m scripts.smoke

if [ ! -f data/processed/spending.parquet ]; then
  echo "== 4/4 SKIPPED: data/processed/spending.parquet not present (private Wallet export)."
  echo "   Training and the evaluation protocol need it; see data/README.md."
  exit 0
fi

echo "== 4/4 full pipeline: train -> export -> eval -> ablation"
uv run python -m selena_ml.train --out artifacts/repro --version repro
uv run python -m scripts.export_onnx --artifact artifacts/repro
uv run python -m scripts.eval_protocol --artifact artifacts/repro --out reports/repro
uv run python -m scripts.ablation
echo "done: reports/repro/eval.md"
