"""Two non-neural baselines. They exist so "the autoencoder works" can be tested, not assumed."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def zscore_scores(X_train: np.ndarray, X_eval: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + eps
    return np.abs((X_eval - mu) / sd).max(axis=1)


def isolation_forest_scores(
    X_train: np.ndarray, X_eval: np.ndarray, seed: int = 42
) -> np.ndarray:
    model = IsolationForest(random_state=seed, n_estimators=200, contamination=0.01)
    model.fit(X_train)
    # higher score = more anomalous
    return -model.score_samples(X_eval)
