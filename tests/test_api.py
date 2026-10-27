import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from selena_ml.data.synthetic import make_synthetic

CATEGORIES = ["Food & Drinks", "Groceries", "Coffee", "Transportation"]


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    from scripts import export_onnx
    from selena_ml import train as train_mod

    tmp = tmp_path_factory.mktemp("art")
    df = make_synthetic(n=260, seed=33)
    rng = np.random.default_rng(3)
    df["category"] = rng.choice(CATEGORIES, size=len(df))
    data = tmp / "tx.parquet"
    df.to_parquet(data)

    out = tmp / "art"
    old = sys.argv
    try:
        sys.argv = ["train", "--data", str(data), "--out", str(out), "--epochs", "4"]
        train_mod.main()
        sys.argv = ["export", "--artifact", str(out)]
        export_onnx.main()
    finally:
        sys.argv = old

    os.environ["SELENA_ARTIFACT_DIR"] = str(out)
    from selena_ml.serve import app as app_mod

    app_mod.ARTIFACT_DIR = out
    app_mod._bundle.cache_clear()
    return TestClient(app_mod.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["detector_default"] == "zscore"


def test_score_returns_one_row_per_input_in_order(client):
    payload = {
        "transactions": [
            {"date": "2026-01-01", "amount": 20000, "category": "Coffee"},
            {"date": "2026-01-02", "amount": 900000, "category": "Groceries"},
        ]
    }
    r = client.post("/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["detector"] == "zscore"
    assert len(body["results"]) == 2
    assert body["results"][0]["date"] == "2026-01-01"
    assert body["flagged"] == sum(x["is_anomaly"] for x in body["results"])


def test_autoencoder_detector_is_available(client):
    payload = {
        "transactions": [{"date": "2026-01-01", "amount": 30000}],
        "detector": "autoencoder",
    }
    r = client.post("/score", json=payload)
    assert r.status_code == 200
    assert r.json()["detector"] == "autoencoder"


def test_empty_list_rejected(client):
    assert client.post("/score", json={"transactions": []}).status_code == 422


def test_negative_amount_rejected(client):
    r = client.post("/score", json={"transactions": [{"date": "2026-01-01", "amount": -1}]})
    assert r.status_code == 422


def test_unknown_detector_rejected(client):
    payload = {"transactions": [{"date": "2026-01-01", "amount": 1000}], "detector": "magic"}
    assert client.post("/score", json=payload).status_code == 422


def test_sample_endpoint_serves_synthetic_rows(client):
    r = client.get("/sample")
    assert r.status_code == 200
    rows = r.json()["transactions"]
    assert len(rows) == 60
    assert {"date", "amount"} <= set(rows[0])


def test_metrics_reports_offline_numbers(client):
    body = client.get("/metrics").json()
    assert body["n_train"] > 0
    assert "autoencoder" in body["metrics"]
