"""MinMax scaler that serialises to JSON (no pickle).

`transform` clips to [0, 1]: values outside the training range (common at inference) do not blow up,
but still show up as a large reconstruction error.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class MinMaxJSON:
    def __init__(self, mins, maxs):
        self.mins = np.asarray(mins, dtype=float)
        self.maxs = np.asarray(maxs, dtype=float)
        if self.mins.shape != self.maxs.shape:
            raise ValueError("mins and maxs must have the same shape")

    @classmethod
    def fit(cls, X) -> MinMaxJSON:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        return cls(X.min(axis=0), X.max(axis=0))

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        span = np.where((self.maxs - self.mins) == 0, 1.0, self.maxs - self.mins)
        return np.clip((X - self.mins) / span, 0.0, 1.0)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"mins": self.mins.tolist(), "maxs": self.maxs.tolist()}, indent=2)
        )

    @classmethod
    def from_json(cls, path: str | Path) -> MinMaxJSON:
        raw = json.loads(Path(path).read_text())
        return cls(raw["mins"], raw["maxs"])
