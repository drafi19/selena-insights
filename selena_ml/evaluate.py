"""Evaluation metrics. Every number in the README must come out of these functions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def metrics_at_threshold(y_true, scores, threshold: float) -> dict:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pred = (s > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    negatives = int((y == 0).sum())
    out = {
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "flagged": int(pred.sum()),
        "n": int(len(y)),
        "fpr": float(pred[y == 0].mean()) if negatives else None,
    }
    if 0 < int(y.sum()) < len(y):
        out["pr_auc"] = float(average_precision_score(y, s))
        out["roc_auc"] = float(roc_auc_score(y, s))
    else:
        out["pr_auc"] = None
        out["roc_auc"] = None
    return out


def precision_at_k(y_true, scores, k: int) -> float:
    y = np.asarray(y_true).astype(int)
    if k <= 0 or len(y) == 0:
        return 0.0
    top = np.argsort(-np.asarray(scores, dtype=float))[:k]
    return float(y[top].mean())


def recall_at_k(y_true, scores, k: int) -> float:
    y = np.asarray(y_true).astype(int)
    total = int(y.sum())
    if k <= 0 or total == 0:
        return 0.0
    top = np.argsort(-np.asarray(scores, dtype=float))[:k]
    return float(y[top].sum() / total)
