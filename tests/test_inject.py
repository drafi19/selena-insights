import numpy as np
import pytest

from selena_ml.data.inject import inject_anomalies
from selena_ml.data.synthetic import make_synthetic


def test_label_count_matches_rate():
    df = make_synthetic(n=500, seed=11)
    _, labels = inject_anomalies(df, rate=0.02, seed=1)
    assert labels["is_injected"].sum() == 10
    assert len(labels) == len(df)


def test_deterministic_across_calls():
    df = make_synthetic(n=300, seed=2)
    a, la = inject_anomalies(df, rate=0.05, seed=9)
    b, lb = inject_anomalies(df, rate=0.05, seed=9)
    assert a.equals(b) and la.equals(lb)


def test_spike_increases_amount():
    df = make_synthetic(n=200, seed=4)
    out, labels = inject_anomalies(df, rate=0.02, seed=3, types=("spike",))
    idx = np.flatnonzero(labels["is_injected"].to_numpy())
    assert (out.loc[idx, "amount"].to_numpy() > df.loc[idx, "amount"].to_numpy()).all()


def test_drip_is_small():
    df = make_synthetic(n=200, seed=4)
    out, labels = inject_anomalies(df, rate=0.02, seed=3, types=("drip",))
    idx = np.flatnonzero(labels["is_injected"].to_numpy())
    assert (out.loc[idx, "amount"].to_numpy() < 1000.0).all()


def test_rejects_bad_rate():
    df = make_synthetic(n=50, seed=1)
    with pytest.raises(ValueError):
        inject_anomalies(df, rate=1.5)
