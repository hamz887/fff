# -*- coding: utf-8 -*-
"""
models/kan.py — Kolmogorov-Arnold Network layers and classifiers (Liu et al., 2024).

Shapes: every model maps x ∈ R^{B×d} to (logits, reconstruction):
  logits ∈ R^{B} for binary output (n_out = 1) or R^{B×n_out} for the softmax attack-type head;
  reconstruction ∈ R^{B×d} (KAN-AE in training mode only) or None.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """Kolmogorov-Arnold layer: y = W_b·SiLU(x) + Σ c·B(x) with learnable B-spline coefficients.

    The uniform knot vector covers ``gr`` extended by k knots on each side; B-splines are evaluated with the
    vectorised Cox-de Boor recursion (x: B×i -> bases: B×i×(G+k))."""

    def __init__(self, i, o, G=5, k=3, gr=(-1.0, 1.0)):
        super().__init__()
        self.i, self.o, self.G, self.k = i, o, G, k
        h = (gr[1] - gr[0]) / G
        self.register_buffer("grid", (torch.arange(-k, G + k + 1, dtype=torch.float32) * h + gr[0]).expand(i, -1)
                             .contiguous())
        self.base_weight = nn.Parameter(torch.empty(o, i))
        self.spline_weight = nn.Parameter(torch.empty(o, i, G + k))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, std=0.1 / math.sqrt(i))

    def b_splines(self, x):
        g, x = self.grid, x.unsqueeze(-1)                              # g: i×(G+2k+1), x: B×i×1
        b = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)            # order-0 bases: B×i×(G+2k)
        for p in range(1, self.k + 1):
            b = ((x - g[:, :-(p + 1)]) / (g[:, p:-1] - g[:, :-(p + 1)]) * b[..., :-1] +
                 (g[:, p + 1:] - x) / (g[:, p + 1:] - g[:, 1:-p]) * b[..., 1:])
        return b                                                       # B×i×(G+k)

    def forward(self, x):
        if x.dim() != 2 or x.size(1) != self.i:
            raise ValueError(f"KANLinear expected input of shape (B, {self.i}), got {tuple(x.shape)}")
        return F.linear(F.silu(x), self.base_weight) + \
            F.linear(self.b_splines(x).reshape(x.size(0), -1), self.spline_weight.reshape(self.o, -1))


class KANClassifier(nn.Module):
    """d -> hidden (KAN) -> LayerNorm -> n_out (KAN, grid [-3, 3] to cover the normalised activations)."""

    def __init__(self, d, hidden=16, G=5, k=3, n_out=1):
        super().__init__()
        self.n_out = n_out
        self.l1 = KANLinear(d, hidden, G, k)
        self.norm = nn.LayerNorm(hidden)
        self.l2 = KANLinear(hidden, n_out, G, k, (-3.0, 3.0))

    def head(self):
        return self.l2

    def forward(self, x):
        z = self.l2(self.norm(self.l1(x)))
        return (z.squeeze(-1) if self.n_out == 1 else z), None


class KANAutoencoderClassifier(nn.Module):
    """Hybrid KAN-AE. Encoder d -> hidden -> latent (tanh keeps the latent inside the spline grid [-1, 1]);
    classifier head on the latent; decoder latent -> hidden -> d used only for the reconstruction loss during
    training (never executed at inference). latent = max(2, ceil(d·ratio)) < d and hidden <= d."""

    def __init__(self, d, latent, G=5, k=3, n_out=1):
        super().__init__()
        if not 1 <= latent < d:
            raise ValueError(f"latent ({latent}) must be in [1, d) with d = {d}")
        hidden = min(max(latent + 1, int(round((d + latent) / 2))), d)
        self.dims, self.n_out = (d, hidden, latent), n_out
        self.enc1, self.enc2 = KANLinear(d, hidden, G, k), KANLinear(hidden, latent, G, k)
        self.dec1, self.dec2 = KANLinear(latent, hidden, G, k), KANLinear(hidden, d, G, k)
        self.cls = KANLinear(latent, n_out, G, k)

    def head(self):
        return self.cls

    def encode(self, x):
        return torch.tanh(self.enc2(torch.tanh(self.enc1(x))))

    def decode(self, z):
        return self.dec2(torch.tanh(self.dec1(z)))

    def forward(self, x, reconstruct=None):
        """reconstruct=None -> decoder runs only in training mode; True/False forces it on/off."""
        z = self.encode(x)
        out = self.cls(z)
        out = out.squeeze(-1) if self.n_out == 1 else out
        rec = self.training if reconstruct is None else reconstruct
        return out, (self.decode(z) if rec else None)
