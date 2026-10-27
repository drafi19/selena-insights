import torch

from selena_ml.models.autoencoder import AutoEncoder


def test_output_shape_matches_input():
    model = AutoEncoder(n_features=6)
    x = torch.rand(11, 6)
    assert model(x).shape == (11, 6)


def test_parameter_count_matches_original_keras_model():
    assert sum(p.numel() for p in AutoEncoder(n_features=6).parameters()) == 1774


def test_reconstruction_error_is_per_row():
    model = AutoEncoder(n_features=6)
    err = model.reconstruction_error(torch.rand(7, 6))
    assert err.shape == (7,)
    assert (err >= 0).all()
