"""One-command smoke test: load the committed artefact, score the demo sample, print the verdict.

Needs no private data - this is what a reviewer runs first.
    uv run python -m scripts.smoke
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault("SELENA_ARTIFACT_DIR", "artifacts/v0.3.0")


def main() -> None:
    from selena_ml.serve import app as app_mod

    client = TestClient(app_mod.app)

    health = client.get("/health").json()
    print("health:", json.dumps(health))

    sample = json.loads(Path("web/sample.json").read_text())["transactions"]
    body = client.post("/score", json={"transactions": sample[:10]}).json()
    flagged = sum(1 for r in body["results"] if r["is_anomaly"])
    print(f"scored {len(body['results'])} demo rows with detector={body['detector']}, "
          f"threshold={body['threshold']:.3f}, flagged={flagged}")
    print("smoke ok")


if __name__ == "__main__":
    main()
