import json
import sys

import pytest

from selena_ml.config import FEATURES, SCORED_FEATURES
from selena_ml.data.synthetic import make_synthetic


def test_scored_features_are_a_subset_and_exclude_the_gap():
    assert set(SCORED_FEATURES) <= set(FEATURES)
    assert "days_since_prev" not in SCORED_FEATURES


@pytest.mark.slow
def test_training_writes_scored_features_in_order(tmp_path):
    from selena_ml import train as train_mod

    data = tmp_path / "tx.parquet"
    make_synthetic(n=160, seed=21).to_parquet(data)
    out = tmp_path / "art"

    old = sys.argv
    sys.argv = [
        "train",
        "--data",
        str(data),
        "--out",
        str(out),
        "--epochs",
        "2",
        "--patience",
        "1",
    ]
    try:
        train_mod.main()
    finally:
        sys.argv = old

    zscore = json.loads((out / "zscore.json").read_text())
    assert zscore["features"] == SCORED_FEATURES
    assert len(zscore["mu"]) == len(SCORED_FEATURES)
    assert len(zscore["sd"]) == len(SCORED_FEATURES)

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["features"] == FEATURES
    assert meta["scored_features"] == SCORED_FEATURES
