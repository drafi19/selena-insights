import numpy as np

from selena_ml.evaluate import metrics_at_threshold, precision_at_k, recall_at_k


def test_perfect_separation_gives_max_scores():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8])
    m = metrics_at_threshold(y, scores, threshold=0.5)
    assert m["precision"] == 1.0 and m["recall"] == 1.0
    assert m["pr_auc"] == 1.0 and m["roc_auc"] == 1.0
    assert m["fpr"] == 0.0


def test_precision_at_k_counts_hits():
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    assert precision_at_k(y, scores, 2) == 0.5
    assert recall_at_k(y, scores, 2) == 0.5


def test_degenerate_labels_return_none_auc():
    assert metrics_at_threshold(np.zeros(5), np.arange(5.0), 2.0)["pr_auc"] is None
