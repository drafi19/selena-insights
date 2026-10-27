import numpy as np

from selena_ml.baselines import isolation_forest_scores, zscore_scores


def test_zscore_flags_obvious_outlier():
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, size=(200, 3))
    ev = np.vstack([rng.normal(0, 1, size=(20, 3)), np.array([[9.0, 9.0, 9.0]])])
    scores = zscore_scores(train, ev)
    assert int(np.argmax(scores)) == 20


def test_iforest_returns_one_score_per_row():
    rng = np.random.default_rng(1)
    train = rng.normal(0, 1, size=(100, 4))
    ev = rng.normal(0, 1, size=(15, 4))
    assert isolation_forest_scores(train, ev).shape == (15,)


def test_iforest_deterministic():
    rng = np.random.default_rng(2)
    train = rng.normal(0, 1, size=(100, 4))
    ev = rng.normal(0, 1, size=(10, 4))
    a = isolation_forest_scores(train, ev, seed=7)
    b = isolation_forest_scores(train, ev, seed=7)
    assert np.allclose(a, b)
