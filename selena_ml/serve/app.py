"""FastAPI service. Only needs onnxruntime - torch is not imported here.

The default detector comes from the artefact metadata. It is z-score, because that is what won
on the evidence (see docs/adr/0002-detector-choice.md); the autoencoder is exposed for comparison,
not because it sounds better.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from selena_ml.features import build_features
from selena_ml.scaling import MinMaxJSON
from selena_ml.serve.schema import ScoredTxn, ScoreRequest, ScoreResponse

ARTIFACT_DIR = Path(os.environ.get("SELENA_ARTIFACT_DIR", "artifacts/v0.3.0"))
WEB_DIR = Path(os.environ.get("SELENA_WEB_DIR", "web"))

app = FastAPI(title="Selena Insights", version="0.3.0")


@lru_cache(maxsize=1)
def _bundle() -> tuple[dict, MinMaxJSON, dict, ort.InferenceSession, float]:
    meta = json.loads((ARTIFACT_DIR / "metadata.json").read_text())
    scaler = MinMaxJSON.from_json(ARTIFACT_DIR / "scaler.json")
    zscore = json.loads((ARTIFACT_DIR / "zscore.json").read_text())
    session = ort.InferenceSession(
        str(ARTIFACT_DIR / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    threshold = float(json.loads((ARTIFACT_DIR / "threshold.json").read_text())["threshold"])
    return meta, scaler, zscore, session, threshold


def _matrix(payloads) -> np.ndarray:
    df = pd.DataFrame(
        [
            {
                "date": t.date.isoformat(),
                "amount": t.amount,
                "category": t.category or "unknown",
            }
            for t in payloads
        ]
    )
    return build_features(df).to_numpy()


@app.get("/health")
def health() -> dict:
    meta, _, _, _, _ = _bundle()
    return {
        "status": "ok",
        "model_version": meta["model_version"],
        "detector_default": meta.get("detector_default", "zscore"),
        "features": len(meta["features"]),
    }


@app.get("/metrics")
def metrics() -> dict:
    meta, _, _, _, _ = _bundle()
    return {
        "model_version": meta["model_version"],
        "n_train": meta["n_train"],
        "n_test": meta["n_test"],
        "metrics": meta.get("metrics"),
    }


@app.get("/sample")
def sample() -> dict:
    path = WEB_DIR / "sample.json"
    if not path.exists():
        raise FileNotFoundError("web/sample.json missing - run scripts.make_demo_sample")
    return json.loads(path.read_text())


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    meta, scaler, zscore, session, thr_ae = _bundle()
    detector = req.detector or meta.get("detector_default", "zscore")

    X = _matrix(req.transactions)
    S = scaler.transform(X)

    if detector == "zscore":
        mu = np.asarray(zscore["mu"], dtype=float)
        sd = np.asarray(zscore["sd"], dtype=float)
        scores = np.abs((S - mu) / sd).max(axis=1)
        threshold = float(zscore["threshold"])
    elif detector == "autoencoder":
        reconstruction = session.run(None, {"input": S.astype("float32")})[0]
        scores = ((S - reconstruction) ** 2).mean(axis=1)
        threshold = thr_ae
    else:
        raise ValueError(f"unknown detector: {detector}")

    results = [
        ScoredTxn(
            date=t.date,
            amount=t.amount,
            category=t.category,
            score=float(scores[i]),
            is_anomaly=bool(scores[i] > threshold),
        )
        for i, t in enumerate(req.transactions)
    ]
    return ScoreResponse(
        model_version=meta["model_version"],
        detector=detector,
        threshold=threshold,
        flagged=sum(r.is_anomaly for r in results),
        results=results,
    )


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
