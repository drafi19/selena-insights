import numpy as np
import pandas as pd
import pytest

from selena_ml.config import FEATURES
from selena_ml.data.synthetic import make_synthetic
from selena_ml.features import build_features


def test_shape_and_columns():
    df = make_synthetic(n=100, seed=1)
    feats = build_features(df)
    assert list(feats.columns) == FEATURES
    assert len(feats) == 100
    assert not feats.isna().any().any()


def test_row_order_preserved():
    df = make_synthetic(n=60, seed=2).sample(frac=1.0, random_state=0).reset_index(drop=True)
    feats = build_features(df)
    smallest = int(np.argmin(df["amount"].to_numpy()))
    assert feats.loc[smallest, "log_amount"] == pytest.approx(np.log1p(df.loc[smallest, "amount"]))


def test_no_future_leakage():
    df = make_synthetic(n=120, seed=3)
    full = build_features(df)
    head = build_features(df.iloc[:60])
    pd.testing.assert_frame_equal(full.iloc[:60], head)


def test_gap_is_clipped():
    df = pd.DataFrame({"date": ["2020-01-01", "2024-01-01"], "amount": [1000.0, 1000.0]})
    feats = build_features(df)
    assert feats.loc[1, "days_since_prev"] == 60.0


def test_missing_column_raises():
    with pytest.raises(ValueError, match="missing required columns"):
        build_features(pd.DataFrame({"amount": [1.0]}))
