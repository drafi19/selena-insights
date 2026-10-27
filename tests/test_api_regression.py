import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

GOLDEN = Path("tests/golden_scores.json")
ARTIFACT = Path("artifacts/v0.3.0")


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden scores not frozen yet")
def test_scores_are_stable():
    os.environ["SELENA_ARTIFACT_DIR"] = str(ARTIFACT)
    from selena_ml.serve import app as app_mod

    app_mod.ARTIFACT_DIR = ARTIFACT
    app_mod._bundle.cache_clear()
    client = TestClient(app_mod.app)

    golden = json.loads(GOLDEN.read_text())
    body = client.post("/score", json={"transactions": golden["input"]}).json()
    assert body["detector"] == golden["detector"]
    assert body["threshold"] == pytest.approx(golden["threshold"])
    got = [round(r["score"], 8) for r in body["results"]]
    for a, b in zip(got, golden["scores"], strict=True):
        assert a == pytest.approx(b, abs=1e-6)
