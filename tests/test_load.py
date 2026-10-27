import pandas as pd
import pytest

from selena_ml.data.load import load_transactions


def test_load_sorted(tmp_path):
    src = pd.DataFrame({"date": ["2026-02-01", "2026-01-01"], "amount": [2000.0, 1000.0]})
    p = tmp_path / "t.parquet"
    src.to_parquet(p)
    out = load_transactions(p)
    assert out["date"].is_monotonic_increasing
    assert out["amount"].tolist() == [1000.0, 2000.0]


def test_missing_column_raises(tmp_path):
    p = tmp_path / "bad.parquet"
    pd.DataFrame({"date": ["2026-01-01"]}).to_parquet(p)
    with pytest.raises(ValueError, match="missing required columns"):
        load_transactions(p)


def test_non_positive_amount_raises(tmp_path):
    p = tmp_path / "neg.parquet"
    pd.DataFrame({"date": ["2026-01-01"], "amount": [-5.0]}).to_parquet(p)
    with pytest.raises(ValueError, match="non-positive"):
        load_transactions(p)
