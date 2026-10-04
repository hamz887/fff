# -*- coding: utf-8 -*-
"""models/mlp.py — MLP baseline whose width is matched to the KAN parameter count (see models.build_model)."""
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, d, width, n_out=1):
        super().__init__()
        self.n_out = n_out
        half = max(width // 2, 4)
        self.net = nn.Sequential(nn.Linear(d, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(width, half), nn.ReLU(), nn.Linear(half, n_out))

    def head(self):
        return self.net[-1]

    def forward(self, x):
        z = self.net(x)
        return (z.squeeze(-1) if self.n_out == 1 else z), None
