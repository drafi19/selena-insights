from selena_ml.data.synthetic import make_synthetic, simulate_transactions


def test_shape_and_columns():
    df = make_synthetic(n=120, seed=1)
    assert len(df) == 120
    assert list(df.columns) == ["date", "amount", "direction"]


def test_deterministic():
    a = make_synthetic(n=50, seed=3)
    b = make_synthetic(n=50, seed=3)
    assert a.equals(b)


def test_amounts_positive_and_rounded():
    df = make_synthetic(n=200, seed=5)
    assert (df["amount"] > 0).all()
    assert ((df["amount"] % 1000) == 0).all()


def test_simulator_deterministic_and_sized():
    a = simulate_transactions(n=500, seed=4)
    b = simulate_transactions(n=500, seed=4)
    assert len(a) == 500
    assert a.equals(b)
    assert list(a.columns) == ["date", "amount", "direction"]


def test_simulator_has_structure_and_sane_scale():
    df = simulate_transactions(n=4000, seed=5)
    is_weekend = df["date"].dt.dayofweek >= 5
    # real data shows a weekend dip, so the simulator must reproduce the direction
    assert df.loc[is_weekend, "amount"].median() < df.loc[~is_weekend, "amount"].median()
    median = df["amount"].median()
    assert 5_000 < median < 60_000            # audit 2026-10-27: median 12k
    share_income = (df["direction"] == "income").mean()
    assert 0.02 < share_income < 0.15
    assert df["date"].is_monotonic_increasing
