"""Measure the z-score feature ablation on the same split and injection seeds.

Run from the repo root:
    uv run python -m scripts.score_ablation --artifact artifacts/v0.4.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from selena_ml.baselines import zscore_scores
from selena_ml.config import EVAL_REPLICATES, FEATURES, INJECT_RATE, SCORED_FEATURES
from selena_ml.data.inject import inject_anomalies
from selena_ml.data.load import load_transactions
from selena_ml.evaluate import metrics_at_threshold
from selena_ml.features import build_features, build_features_with_history
from selena_ml.scaling import MinMaxJSON
from selena_ml.train import split_time

CLEAN_WINDOW_DAYS = 90
TOP_K = 30


def tie_count(scores: np.ndarray, k: int = TOP_K) -> int:
    top = np.sort(np.asarray(scores, dtype=float))[-k:]
    return int(len(top) - len(np.unique(top)))


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/v0.4.0")
    ap.add_argument("--data", default="data/processed/spending.parquet")
    ap.add_argument("--replicates", type=int, default=EVAL_REPLICATES)
    ap.add_argument("--out", default="reports/score_ablation.md")
    args = ap.parse_args()

    art = Path(args.artifact)
    scaler = MinMaxJSON.from_json(art / "scaler.json")
    zscore_path = art / "zscore.json"
    threshold = 3.0
    if zscore_path.exists():
        threshold = float(json.loads(zscore_path.read_text()).get("threshold", threshold))

    schemes = {
        "full7": FEATURES,
        "amount2": SCORED_FEATURES,
        "amount3": ["log_amount", *SCORED_FEATURES],
        "no_gap": [name for name in FEATURES if name != "days_since_prev"],
    }
    index = {name: i for i, name in enumerate(FEATURES)}
    missing = [name for names in schemes.values() for name in names if name not in index]
    if missing:
        raise ValueError(f"scheme features missing from FEATURES: {missing}")
    cols = {name: [index[feature] for feature in names] for name, names in schemes.items()}

    df = load_transactions(args.data)
    train_df, val_df, test_df = split_time(df)

    # warm: one pass over the full series, sliced at the split positions.
    full = build_features(df)
    S_tr = scaler.transform(full.iloc[: len(train_df)].to_numpy())
    history = df.iloc[: len(train_df) + len(val_df)].reset_index(drop=True)

    pooled_y: list[np.ndarray] = []
    warm_scores: dict[str, list[np.ndarray]] = {name: [] for name in schemes}
    cold_scores: dict[str, list[np.ndarray]] = {name: [] for name in schemes}
    for r in range(args.replicates):
        injected, labels = inject_anomalies(test_df, rate=INJECT_RATE, seed=10_000 + r)
        S_inj_warm = scaler.transform(build_features_with_history(history, injected).to_numpy())
        S_inj_cold = scaler.transform(build_features(injected).to_numpy())
        pooled_y.append(labels["is_injected"].to_numpy().astype(int))
        for name, scheme_cols in cols.items():
            warm_scores[name].append(zscore_scores(S_tr, S_inj_warm, scheme_cols))
            cold_scores[name].append(zscore_scores(S_tr, S_inj_cold, scheme_cols))

    yy = np.concatenate(pooled_y)
    pooled = {
        name: metrics_at_threshold(yy, np.concatenate(values), threshold)
        for name, values in warm_scores.items()
    }
    cold_pooled = {
        name: metrics_at_threshold(yy, np.concatenate(values), threshold)
        for name, values in cold_scores.items()
    }

    cutoff = df["date"].max() - pd.Timedelta(days=CLEAN_WINDOW_DAYS)
    recent = df[df["date"] >= cutoff].reset_index(drop=True)
    earlier = df[df["date"] < cutoff].reset_index(drop=True)
    S_clean = scaler.transform(build_features_with_history(earlier, recent).to_numpy())
    ties = {
        name: tie_count(zscore_scores(S_tr, S_clean, scheme_cols))
        for name, scheme_cols in cols.items()
    }

    best = max(pooled, key=lambda name: pooled[name]["pr_auc"] or 0.0)
    lines = [
        "# Score ablation - z-score feature sets (warm features)",
        "",
        f"Artefact `{args.artifact}`. The same real-data split and the same "
        f"{args.replicates} injection seeds for every scheme. Warm features use the real history "
        f"before the test window; cold rebuilds features separately per split, as the old script "
        f"did. PR-AUC and ROC-AUC are pooled across replicates; ties are counted among the {TOP_K} "
        f"highest warm scores on the clean {CLEAN_WINDOW_DAYS}-day window at threshold "
        f"{threshold:.1f}.",
        "",
        "| Scheme | Features | PR-AUC warm | ROC-AUC warm | PR-AUC cold | ROC-AUC cold "
        "| Top-30 ties |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, names in schemes.items():
        lines.append(
            f"| {name} | {', '.join(names)} | {fmt(pooled[name]['pr_auc'])} | "
            f"{fmt(pooled[name]['roc_auc'])} | {fmt(cold_pooled[name]['pr_auc'])} | "
            f"{fmt(cold_pooled[name]['roc_auc'])} | {ties[name]} |"
        )

    lines += [
        "",
        "## Reading",
        "",
        f"- `{best}` has the highest warm pooled PR-AUC ({fmt(pooled[best]['pr_auc'])}).",
        f"- The shipped amount-only score (`amount2`) scores {fmt(pooled['amount2']['pr_auc'])} "
        f"warm against {fmt(pooled['full7']['pr_auc'])} for the old 7-feature score; cold those "
        f"numbers are {fmt(cold_pooled['amount2']['pr_auc'])} and "
        f"{fmt(cold_pooled['full7']['pr_auc'])}.",
        "- `no_gap` removes `days_since_prev` from the full set; `amount3` adds the raw "
        "`log_amount` back to the shipped pair.",
        "- Top-30 ties on the clean window: "
        + ", ".join(f"`{name}` {ties[name]}" for name in schemes)
        + ".",
        "- Ties matter because a long gap clipped at 60 days collapses many rows onto the "
        "same score; they are counted on the clean series only, without injected rows.",
        "",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
