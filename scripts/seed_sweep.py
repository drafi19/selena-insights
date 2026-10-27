"""Retrain with several seeds and report mean +- std of the autoencoder metrics.

Run: uv run python -m scripts.seed_sweep --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        out = Path("artifacts") / f"sweep-seed-{seed}"
        cmd = [
            sys.executable, "-m", "selena_ml.train",
            "--out", str(out), "--epochs", str(args.epochs),
            "--version", f"seed-{seed}", "--seed", str(seed),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        rows.append(json.loads((out / "eval.json").read_text()))

    lines = ["# Seed sweep", "", f"seeds: {args.seeds}", ""]
    for model in ("autoencoder", "zscore", "isolation_forest"):
        prauc = np.array([r[model]["pr_auc"] for r in rows], dtype=float)
        f1 = np.array([r[model]["f1"] for r in rows], dtype=float)
        lines.append(
            f"- **{model}**: PR-AUC {prauc.mean():.3f} +- {prauc.std():.3f}; "
            f"F1 {f1.mean():.3f} +- {f1.std():.3f}"
        )
    Path("reports").mkdir(exist_ok=True)
    Path("reports/eval_seeds.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
