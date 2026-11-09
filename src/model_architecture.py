"""Arsitektur PyTorch untuk SmartClinic No-Show Predictor (klasifikasi biner)."""
from torch import nn


class NoShowMLP(nn.Module):
    """Multi-Layer Perceptron kecil untuk data tabular.

    Keluaran berupa *logit* (belum sigmoid). Probabilitas no-show = sigmoid(logit).
    """

    def __init__(self, input_dim: int, hidden_dims=(64, 32), dropout: float = 0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
