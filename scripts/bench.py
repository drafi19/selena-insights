"""Latency benchmark against the in-process API.

In-process (TestClient), so the number includes request validation and pandas feature building
but not real network/TLS. Run after the artefact exists:

    uv run python -m scripts.bench
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

from selena_ml.serve.app import app

ROWS = 50
REQUESTS = 100


def main() -> None:
    sample = json.loads(Path("web/sample.json").read_text())["transactions"][:ROWS]
    payload = {"transactions": sample}
    client = TestClient(app)

    for _ in range(5):  # warm-up: first call loads the bundle
        client.post("/score", json=payload)

    timings = []
    for _ in range(REQUESTS):
        start = time.perf_counter()
        resp = client.post("/score", json=payload)
        timings.append((time.perf_counter() - start) * 1000)
        if resp.status_code != 200:
            raise SystemExit(f"request failed: {resp.status_code}")

    timings.sort()
    p50 = statistics.median(timings)
    p95 = timings[int(len(timings) * 0.95) - 1]
    print(json.dumps(
        {
            "rows_per_request": len(sample),
            "requests": REQUESTS,
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "mean_ms": round(statistics.fmean(timings), 2),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
