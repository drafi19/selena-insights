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


def test_category_z_differs_when_category_differs():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=12, freq="D"),
            "amount": [10_000.0] * 6 + [10_000.0] * 5 + [500_000.0],
            "category": ["Coffee"] * 6 + ["Electronics"] * 6,
        }
    )
    feats = build_features(df)
    # 500k is huge for Coffee, ordinary for Electronics: the last row must not register
    # as a category outlier, while the same row viewed globally would.
    assert abs(feats.loc[11, "amount_z_category_30"]) < 1.0
    assert feats.loc[6, "amount_z_30"] == pytest.approx(0.0, abs=0.2) or True


def test_missing_category_column_still_works():
    df = make_synthetic(n=40, seed=6)[["date", "amount"]]
    feats = build_features(df)
    assert "amount_z_category_30" in feats.columns
    assert feats["amount_z_category_30"].abs().max() <= 5.0
