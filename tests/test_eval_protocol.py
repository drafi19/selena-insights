import numpy as np
import pandas as pd

from selena_ml.baselines import zscore_scores
from selena_ml.config import FEATURES
from selena_ml.data.synthetic import make_synthetic
from selena_ml.features import build_features, build_features_with_history


def test_spike_scores_higher_with_history_than_cold():
    history = make_synthetic(n=60, seed=21)
    rows = pd.DataFrame(
        {
            "date": pd.date_range("2024-03-01", periods=8, freq="D"),
            "amount": np.round(np.linspace(20_000, 34_000, 8)),
        }
    )
    rows.loc[3, "amount"] = rows.loc[3, "amount"] * 15.0

    S_train = build_features(history).to_numpy()
    cols = [FEATURES.index("amount_z_30")]
    cold = build_features(rows)
    warm = build_features_with_history(history, rows)

    cold_score = zscore_scores(S_train, cold.to_numpy(), cols)
    warm_score = zscore_scores(S_train, warm.to_numpy(), cols)

    assert warm_score[3] > cold_score[3]
