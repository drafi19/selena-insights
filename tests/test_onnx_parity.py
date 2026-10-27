import numpy as np
import onnxruntime as ort
import pytest
import torch

from selena_ml.models.autoencoder import AutoEncoder


def test_onnx_matches_torch(tmp_path):
    pytest.importorskip("onnx")
    model = AutoEncoder(n_features=6)
    model.eval()
    p = tmp_path / "m.onnx"
    torch.onnx.export(
        model,
        torch.rand(3, 6),
        p,
        input_names=["input"],
        output_names=["reconstruction"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(1).random((9, 6)).astype("float32")
    with torch.no_grad():
        a = model(torch.tensor(x)).numpy()
    b = sess.run(None, {"input": x})[0]
    assert np.abs(a - b).max() < 1e-5
