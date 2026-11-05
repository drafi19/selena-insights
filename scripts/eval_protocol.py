"""Official evaluation protocol - every number in reports/eval.md comes out of this file.

A. real test (Wallet data): 20 injection replicates -> mean +- std
A2. pooled across replicates (one number, ~20x the labels)
B. recall per anomaly type
C. synthetic corpus (10k rows, 5% injection) -> statistical power
D. flag rate per category on the CLEAN test set

Run from the repo root:
    uv run python -m scripts.eval_protocol --artifact artifacts/v0.1.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from selena_ml.baselines import isolation_forest_scores, zscore_scores
from selena_ml.config import (
    EVAL_REPLICATES,
    FEATURES,
    INJECT_RATE,
    SEED,
    SYNTH_INJECT_RATE,
    SYNTH_ROWS,
    TARGET_FPR,
)
from selena_ml.data.inject import ANOMALY_TYPES, inject_anomalies
from selena_ml.data.load import load_transactions
from selena_ml.data.synthetic import simulate_transactions
from selena_ml.evaluate import metrics_at_threshold, precision_at_k
from selena_ml.features import build_features, build_features_with_history
from selena_ml.models.autoencoder import AutoEncoder
from selena_ml.scaling import MinMaxJSON
from selena_ml.train import fit_autoencoder, split_time

MODELS = ("autoencoder", "zscore", "isolation_forest")
SYNTH_SEEDS = 3
MIN_CATEGORY_ROWS = 10


def load_artifact(art: Path) -> tuple[AutoEncoder, MinMaxJSON, float, dict, dict]:
    model = AutoEncoder(n_features=len(FEATURES))
    model.load_state_dict(torch.load(art / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    scaler = MinMaxJSON.from_json(art / "scaler.json")
    threshold = json.loads((art / "threshold.json").read_text())["threshold"]
    meta = json.loads((art / "metadata.json").read_text())
    zscore_path = art / "zscore.json"
    zstats = json.loads(zscore_path.read_text()) if zscore_path.exists() else {"threshold": 3.0}
    return model, scaler, float(threshold), meta, zstats


def ae_error(model: AutoEncoder, S: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model.reconstruction_error(torch.tensor(S, dtype=torch.float32)).numpy()


def all_scores(model, S_train, S_eval, zscore_cols=None) -> dict[str, np.ndarray]:
    return {
        "autoencoder": ae_error(model, S_eval),
        "zscore": zscore_scores(S_train, S_eval, zscore_cols),
        "isolation_forest": isolation_forest_scores(S_train, S_eval, SEED),
    }


def mean_std(values) -> str:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f} +- {np.std(vals):.3f}"


def split_feature_frames(df, warm: bool):
    """Return the split frames plus train/val features.

    warm=True builds once on the full series and slices by split position (train is first, so
    its slice equals the old cold build). warm=False keeps the old per-split path for table A0.
    """
    train_df, val_df, test_df = split_time(df)
    if warm:
        full = build_features(df)
        a = len(train_df)
        b = a + len(val_df)
        return (
            train_df,
            val_df,
            test_df,
            full.iloc[:a].reset_index(drop=True),
            full.iloc[a:b].reset_index(drop=True),
        )
    return (
        train_df,
        val_df,
        test_df,
        build_features(train_df),
        build_features(val_df),
    )


def run_real_protocol(
    df,
    model,
    scaler,
    thr_ae: float,
    zscore_threshold: float,
    zscore_cols,
    replicates: int,
    warm: bool,
) -> dict:
    """Run tables A/A2/B on the real test split, warm or through the old cold path."""
    train_df, val_df, test_df, f_tr, f_va = split_feature_frames(df, warm)
    S_tr = scaler.transform(f_tr.to_numpy())
    S_va = scaler.transform(f_va.to_numpy())

    thresholds = {
        "autoencoder": thr_ae,
        "zscore": zscore_threshold,
        "isolation_forest": float(
            np.quantile(isolation_forest_scores(S_tr, S_va, SEED), 1.0 - TARGET_FPR)
        ),
    }
    history = df.iloc[: len(train_df) + len(val_df)].reset_index(drop=True)

    rows: dict[str, list[dict]] = {m: [] for m in MODELS}
    by_type: dict[str, list[float]] = {t: [] for t in ANOMALY_TYPES}
    positives: list[int] = []
    pooled_y: list[np.ndarray] = []
    pooled_scores: dict[str, list[np.ndarray]] = {m: [] for m in MODELS}

    for r in range(replicates):
        injected, labels = inject_anomalies(test_df, rate=INJECT_RATE, seed=10_000 + r)
        if warm:
            feats = build_features_with_history(history, injected)
        else:
            feats = build_features(injected)
        S_inj = scaler.transform(feats.to_numpy())
        scores = all_scores(model, S_tr, S_inj, zscore_cols)
        y = labels["is_injected"].to_numpy().astype(int)
        positives.append(int(y.sum()))
        pooled_y.append(y)
        for name in MODELS:
            pooled_scores[name].append(scores[name])
            row = metrics_at_threshold(y, scores[name], thresholds[name])
            row["precision_at_10"] = precision_at_k(y, scores[name], 10)
            rows[name].append(row)

        flagged = scores["autoencoder"] > thr_ae
        for kind in ANOMALY_TYPES:
            mask = (labels["anomaly_type"] == kind).to_numpy()
            if mask.any():
                by_type[kind].append(float(flagged[mask].mean()))

    yy = np.concatenate(pooled_y)
    pooled: dict[str, dict] = {
        name: metrics_at_threshold(yy, np.concatenate(pooled_scores[name]), thresholds[name])
        for name in MODELS
    }
    return {
        "rows": rows,
        "by_type": by_type,
        "positives": positives,
        "yy": yy,
        "pooled": pooled,
        "thresholds": thresholds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/v0.4.0")
    ap.add_argument("--data", default="data/processed/spending.parquet")
    ap.add_argument("--replicates", type=int, default=EVAL_REPLICATES)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    model, scaler, thr_ae, meta, zstats = load_artifact(Path(args.artifact))
    feature_names = meta.get("features", FEATURES)
    if zstats.get("features"):
        zscore_cols = [feature_names.index(name) for name in zstats["features"]]
    else:
        zscore_cols = None
    zscore_threshold = float(zstats.get("threshold", 3.0))
    df = load_transactions(args.data)
    train_df, _val_df, test_df = split_time(df)

    # Primary result: features are built once over the full series and the injection replicates
    # keep the real history that precedes the test window.
    warm = run_real_protocol(
        df, model, scaler, thr_ae, zscore_threshold, zscore_cols, args.replicates, warm=True
    )
    # Appendix A0: the old per-split cold path, kept runnable so the report can compare both.
    cold = run_real_protocol(
        df, model, scaler, thr_ae, zscore_threshold, zscore_cols, args.replicates, warm=False
    )
    rows = warm["rows"]
    by_type = warm["by_type"]
    positives = warm["positives"]
    yy = warm["yy"]
    pooled = warm["pooled"]
    cold_pooled = cold["pooled"]

    # C trains a SEPARATE model on the synthetic corpus. Applying the real-data model to synthetic
    # rows would only measure domain shift, not how the method behaves with more data.
    synth = simulate_transactions(n=SYNTH_ROWS, seed=SEED + 1)
    syn_train, syn_val, syn_test = split_time(synth)
    syn_scaler = MinMaxJSON.fit(build_features(syn_train).to_numpy())
    Ssyn_tr = syn_scaler.transform(build_features(syn_train).to_numpy())
    Ssyn_va = syn_scaler.transform(build_features(syn_val).to_numpy())

    syn_model, _, _ = fit_autoencoder(
        Ssyn_tr, Ssyn_va, epochs=30, batch_size=64, lr=1e-3, patience=5
    )
    syn_thresholds = {
        "autoencoder": float(np.quantile(ae_error(syn_model, Ssyn_va), 1.0 - TARGET_FPR)),
        "zscore": 3.0,
        "isolation_forest": float(
            np.quantile(isolation_forest_scores(Ssyn_tr, Ssyn_va, SEED), 1.0 - TARGET_FPR)
        ),
    }

    synth_rows: dict[str, list[dict]] = {m: [] for m in MODELS}
    for r in range(SYNTH_SEEDS):
        injected, labels = inject_anomalies(syn_test, rate=SYNTH_INJECT_RATE, seed=500 + r)
        S_inj = syn_scaler.transform(build_features(injected).to_numpy())
        scores = {
            "autoencoder": ae_error(syn_model, S_inj),
            "zscore": zscore_scores(Ssyn_tr, S_inj, zscore_cols),
            "isolation_forest": isolation_forest_scores(Ssyn_tr, S_inj, SEED),
        }
        y = labels["is_injected"].to_numpy().astype(int)
        for name in MODELS:
            synth_rows[name].append(metrics_at_threshold(y, scores[name], syn_thresholds[name]))

    # D scores the whole clean series in one pass: the 38-row test slice cannot support a
    # per-category breakdown, and this is a diagnostic, not a reported metric.
    S_all = scaler.transform(build_features(df).to_numpy())
    clean_flag = ae_error(model, S_all) > thr_ae
    per_category: list[dict] = []
    if "category" in df.columns:
        cat = df["category"].fillna("unknown").to_numpy()
        for name in sorted(set(cat)):
            mask = cat == name
            if mask.sum() < MIN_CATEGORY_ROWS:
                continue
            per_category.append(
                {
                    "category": name,
                    "n": int(mask.sum()),
                    "flag_rate": float(clean_flag[mask].mean()),
                }
            )
    per_category.sort(key=lambda d: (-d["flag_rate"], -d["n"]))

    lines = [
        f"# Evaluation - {meta.get('model_version', Path(args.artifact).name)} - warm features",
        "",
        f"Model trained on real Wallet spending (n={len(train_df)}). Test n={len(test_df)}; "
        f"injection {INJECT_RATE:.0%} x {args.replicates} replicates "
        f"(~{int(np.mean(positives))} labels per replicate).",
        "",
        "Features are **warm**: tables A and A2 build features once over the full series and slice",
        "by split position, and every injection replicate uses the real history that precedes the",
        "test window. The cold numbers in appendix A0 rebuild features separately per split, which",
        "restarts the rolling windows at zero. That cold start is an artefact of rebuilding",
        "features per split, not a property of the detector: the served daily scorer keeps context",
        "(`scripts/daily_score.py --context-days`), while the demo page does not. That difference",
        "is why the demo shows few flags.",
        "",
        "## A. Real test set (warm features)",
        "",
        "| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | P@10 | Flagged |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in MODELS:
        rr = rows[name]
        cells = [
            mean_std([x["pr_auc"] for x in rr]),
            mean_std([x["roc_auc"] for x in rr]),
            mean_std([x["precision"] for x in rr]),
            mean_std([x["recall"] for x in rr]),
            mean_std([x["f1"] for x in rr]),
            mean_std([x["precision_at_10"] for x in rr]),
            f"{np.mean([x['flagged'] for x in rr]):.0f}",
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += [
        "",
        f"## A2. Pooled across {args.replicates} replicates (~{len(yy)} labels)",
        "",
        "One number per model, computed from scores and labels concatenated across replicates.",
        "This is the number quoted in the README; table A shows the spread.",
        "",
        "| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for name in MODELS:
        pp = pooled[name]
        lines.append(
            f"| {name} | {mean_std([pp['pr_auc']])} | {mean_std([pp['roc_auc']])} | "
            f"{mean_std([pp['precision']])} | {mean_std([pp['recall']])} | "
            f"{mean_std([pp['f1']])} |"
        )

    lines += [
        "",
        "## A0. Cold-start baseline (old path, not primary)",
        "",
        "The same detectors and seeds, but features are rebuilt separately on train, val, test and",
        "every injected test frame, so the rolling windows start cold. This is the old number set;",
        "compare it with the warm numbers in A2.",
        "",
        "| Model | PR-AUC | ROC-AUC |",
        "|---|---|---|",
    ]
    for name in MODELS:
        cp = cold_pooled[name]
        lines.append(
            f"| {name} | {mean_std([cp['pr_auc']])} | {mean_std([cp['roc_auc']])} |"
        )

    lines += [
        "",
        "## B. Recall per anomaly type (autoencoder)",
        "",
        "| Type | Recall |",
        "|---|---|",
    ]
    for kind in ANOMALY_TYPES:
        lines.append(f"| {kind} | {mean_std(by_type[kind])} |")

    lines += [
        "",
        f"## C. Synthetic corpus ({SYNTH_ROWS:,} rows, {SYNTH_INJECT_RATE:.0%} injection)",
        "",
        "A separate model is trained on this corpus; the real-data model is not transferred here.",
        "",
        "| Model | PR-AUC | Precision | Recall | F1 |",
        "|---|---|---|---|---|",
    ]
    for name in MODELS:
        rr = synth_rows[name]
        cells = [
            mean_std([x["pr_auc"] for x in rr]),
            mean_std([x["precision"] for x in rr]),
            mean_std([x["recall"] for x in rr]),
            mean_std([x["f1"] for x in rr]),
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## D. Flag rate per category (clean full series)",
        "",
        "| Category | n | Flag rate |",
        "|---|---|---|",
    ]
    for d in per_category:
        lines.append(f"| {d['category']} | {d['n']} | {d['flag_rate']:.1%} |")
    lines.append("")
    overall = f"Overall flag rate on the clean full series: **{clean_flag.mean():.1%}**"
    lines.append(f"{overall} (target FPR 1%).")
    lines.append(
        "A category far above the average means the model punishes that category, not a row."
    )

    lines += [
        "",
        "## Limitations",
        "- Labels come from synthetic injection, not human judgement. These numbers measure",
        "  sensitivity to the patterns we defined, not every kind of fraud.",
        "- Table A reuses one trained model across replicates; the variance comes from",
        "  which rows were mutated, not from the weights (that is the seed sweep).",
        "- Table C must never be averaged together with table A.",
        "- Human-labelled precision lives in `reports/manual_review.md` (T25b).",
    ]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.md").write_text("\n".join(lines) + "\n")
    (out_dir / "eval.json").write_text(
        json.dumps(
            {
                "features": "warm",
                "real": rows,
                "pooled": pooled,
                "cold_pooled": cold_pooled,
                "by_type": by_type,
                "synthetic": synth_rows,
                "per_category": per_category,
            },
            indent=2,
        )
    )
    print("\n".join(lines))
    print(f"\n-> {out_dir / 'eval.md'}")


if __name__ == "__main__":
    main()
