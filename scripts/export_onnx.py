"""Export model.pt to ONNX and verify parity against torch.

Run: uv run python -m scripts.export_onnx --artifact artifacts/v0.3.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from selena_ml.config import FEATURES
from selena_ml.models.autoencoder import AutoEncoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/v0.3.0")
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args()

    art = Path(args.artifact)
    model = AutoEncoder(n_features=len(FEATURES))
    model.load_state_dict(torch.load(art / "model.pt", map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.rand(4, len(FEATURES))
    onnx_path = art / "model.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["reconstruction"],
        dynamic_axes={"input": {0: "batch"}, "reconstruction": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).random((32, len(FEATURES))).astype("float32")
    with torch.no_grad():
        torch_out = model(torch.tensor(x)).numpy()
    onnx_out = sess.run(None, {"input": x})[0]
    diff = float(np.abs(torch_out - onnx_out).max())
    if diff > args.atol:
        raise SystemExit(f"parity failed: max diff {diff} > {args.atol}")
    print(json.dumps({"max_abs_diff": diff, "onnx": str(onnx_path)}, indent=2))


if __name__ == "__main__":
    main()
