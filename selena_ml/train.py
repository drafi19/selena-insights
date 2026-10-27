"""Train the autoencoder, calibrate the threshold, compare with baselines, write the bundle.

Run from the repo root:
    uv run python -m selena_ml.train --out artifacts/v0.1.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from selena_ml.artifacts import sha256_file, write_metadata
from selena_ml.baselines import isolation_forest_scores, zscore_scores
from selena_ml.config import FEATURES, INJECT_RATE, SEED, TARGET_FPR, TIME_SPLIT
from selena_ml.data.inject import inject_anomalies
from selena_ml.data.load import load_transactions
from selena_ml.evaluate import metrics_at_threshold, precision_at_k, recall_at_k
from selena_ml.features import build_features
from selena_ml.models.autoencoder import AutoEncoder
from selena_ml.scaling import MinMaxJSON


def split_time(
    df: pd.DataFrame, frac: tuple[float, float, float] = TIME_SPLIT
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    a = int(n * frac[0])
    b = int(n * (frac[0] + frac[1]))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def fit_autoencoder(
    X: np.ndarray,
    X_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 20,
) -> tuple[AutoEncoder, float, int]:
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    model = AutoEncoder(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    xt = torch.tensor(X, dtype=torch.float32)
    xv = torch.tensor(X_val, dtype=torch.float32)

    best_loss, best_state, wait, ran = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xt))
        for start in range(0, len(xt), batch_size):
            batch = xt[perm[start : start + batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv), xv))
        ran = epoch
        if val_loss < best_loss - 1e-8:
            best_loss, wait = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("training never produced a valid state")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_loss, ran


def ae_error(model: AutoEncoder, S: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model.reconstruction_error(torch.tensor(S, dtype=torch.float32)).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/spending.parquet")
    ap.add_argument("--out", default="artifacts/v0.1.0")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--version", default="v0.1.0")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_transactions(args.data)
    train_df, val_df, test_df = split_time(df)
    if min(len(train_df), len(val_df), len(test_df)) < 20:
        raise ValueError("split too small; collect more data")

    X_tr = build_features(train_df).to_numpy()
    X_va = build_features(val_df).to_numpy()

    scaler = MinMaxJSON.fit(X_tr)
    S_tr = scaler.transform(X_tr)
    S_va = scaler.transform(X_va)

    model, best_val_loss, epochs_run = fit_autoencoder(
        S_tr, S_va, args.epochs, args.batch_size, args.lr, args.patience
    )

    err_va = ae_error(model, S_va)

    # threshold from the CLEAN holdout at the target FPR
    threshold = float(np.quantile(err_va, 1.0 - TARGET_FPR))
    thr_if = float(np.quantile(isolation_forest_scores(S_tr, S_va, SEED), 1.0 - TARGET_FPR))

    test_injected, labels = inject_anomalies(test_df, rate=INJECT_RATE, seed=SEED)
    S_inj = scaler.transform(build_features(test_injected).to_numpy())
    err_inj = ae_error(model, S_inj)

    y = labels["is_injected"].to_numpy().astype(int)
    results = {
        "autoencoder": metrics_at_threshold(y, err_inj, threshold),
        "zscore": metrics_at_threshold(y, zscore_scores(S_tr, S_inj), 3.0),
        "isolation_forest": metrics_at_threshold(
            y, isolation_forest_scores(S_tr, S_inj, SEED), thr_if
        ),
    }
    for name, scores in (
        ("autoencoder", err_inj),
        ("zscore", zscore_scores(S_tr, S_inj)),
        ("isolation_forest", isolation_forest_scores(S_tr, S_inj, SEED)),
    ):
        results[name]["precision_at_10"] = precision_at_k(y, scores, 10)
        results[name]["recall_at_10"] = recall_at_k(y, scores, 10)

    torch.save(model.state_dict(), out / "model.pt")
    scaler.to_json(out / "scaler.json")
    # the z-score detector needs the training statistics at inference time
    (out / "zscore.json").write_text(
        json.dumps(
            {
                "mu": S_tr.mean(axis=0).tolist(),
                "sd": (S_tr.std(axis=0) + 1e-6).tolist(),
                "threshold": 3.0,
            },
            indent=2,
        )
    )
    (out / "threshold.json").write_text(
        json.dumps({"threshold": threshold, "target_fpr": TARGET_FPR}, indent=2)
    )
    (out / "eval.json").write_text(json.dumps(results, indent=2))

    meta = {
        "model_version": args.version,
        "features": FEATURES,
        "hidden": [32, 16, 8],
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "seed": SEED,
        "epochs_run": epochs_run,
        "best_val_loss": best_val_loss,
        "threshold": threshold,
        "target_fpr": TARGET_FPR,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_injected": int(y.sum()),
        "detector_default": "zscore",
        "data_sha256": sha256_file(args.data),
        "metrics": results,
    }
    write_metadata(out, meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "metrics"}, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
