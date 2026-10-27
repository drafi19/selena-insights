"""MLP autoencoder with the same architecture as the original Keras model.

6 -> 32 -> 16 -> 8 -> 16 -> 32 -> 6, ReLU, sigmoid output. 1,774 parameters.
"""
from __future__ import annotations

import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, n_features: int = 6, hidden: tuple[int, ...] = (32, 16, 8)):
        super().__init__()
        encoder: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            encoder += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        decoder: list[nn.Module] = []
        for h in reversed(hidden[:-1]):
            decoder += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        decoder += [nn.Linear(prev, n_features), nn.Sigmoid()]
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.forward(x)) ** 2).mean(dim=1)
