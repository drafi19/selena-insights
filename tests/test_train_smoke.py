import json
import sys

import pytest

from selena_ml.data.synthetic import make_synthetic


@pytest.mark.slow
def test_training_writes_bundle(tmp_path):
    from selena_ml import train as train_mod

    data = tmp_path / "tx.parquet"
    make_synthetic(n=300, seed=21).to_parquet(data)

    out = tmp_path / "art"
    argv = ["--data", str(data), "--out", str(out), "--epochs", "5", "--patience", "2"]
    old = sys.argv
    sys.argv = ["train", *argv]
    try:
        train_mod.main()
    finally:
        sys.argv = old

    for name in ("model.pt", "scaler.json", "threshold.json", "eval.json", "metadata.json"):
        assert (out / name).exists(), name
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["threshold"] > 0
    assert meta["n_injected"] > 0
    assert meta["parameters"] == 1839  # 7 features (v0.2); 1774 was the 6-feature model
