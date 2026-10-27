"""Freeze a handful of scores from the shipped artefact for the regression test.

Run once after training, and again whenever the model intentionally changes:
    uv run python -m scripts.freeze_golden
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

ARTIFACT = Path("artifacts/v0.3.0")


def main() -> None:
    os.environ["SELENA_ARTIFACT_DIR"] = str(ARTIFACT)
    from selena_ml.serve import app as app_mod

    app_mod.ARTIFACT_DIR = ARTIFACT
    app_mod._bundle.cache_clear()
    client = TestClient(app_mod.app)

    sample = json.loads(Path("web/sample.json").read_text())["transactions"][:5]
    resp = client.post("/score", json={"transactions": sample})
    resp.raise_for_status()
    body = resp.json()

    golden = {
        "artifact": str(ARTIFACT),
        "detector": body["detector"],
        "threshold": body["threshold"],
        "input": sample,
        "scores": [round(r["score"], 8) for r in body["results"]],
    }
    out = Path("tests/golden_scores.json")
    out.write_text(json.dumps(golden, indent=2), encoding="utf-8")
    print(f"frozen {len(golden['scores'])} scores -> {out}")


if __name__ == "__main__":
    main()
