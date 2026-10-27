import numpy as np

from selena_ml.scaling import MinMaxJSON


def test_fit_transform_range():
    X = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
    s = MinMaxJSON.fit(X)
    out = s.transform(X)
    assert out.min() == 0.0 and out.max() == 1.0
    assert out.shape == X.shape


def test_zero_variance_column_is_safe():
    X = np.array([[3.0, 1.0], [3.0, 2.0]])
    out = MinMaxJSON.fit(X).transform(X)
    assert np.isfinite(out).all()


def test_out_of_range_clipped():
    s = MinMaxJSON.fit(np.array([[0.0], [1.0]]))
    assert s.transform(np.array([[5.0]]))[0, 0] == 1.0


def test_json_roundtrip(tmp_path):
    s = MinMaxJSON.fit(np.array([[0.0, 2.0], [4.0, 6.0]]))
    p = tmp_path / "scaler.json"
    s.to_json(p)
    back = MinMaxJSON.from_json(p)
    x = np.array([[2.0, 4.0]])
    assert np.allclose(s.transform(x), back.transform(x))
