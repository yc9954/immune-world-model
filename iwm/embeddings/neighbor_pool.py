"""Multi-cell mean-field pooling.

Given a query cell state z_i and a set of K neighbor states, produce a
fixed-size a_env that represents the cell's local environment.

Two pools in parallel:
  * simple mean
  * attention-weighted mean (query = z_i)
concatenated and projected.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class NeighborPool(nn.Module):
    def __init__(self, latent_dim: int, env_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.k = nn.Linear(latent_dim, latent_dim, bias=False)
        self.v = nn.Linear(latent_dim, latent_dim, bias=False)
        self.proj = nn.Linear(latent_dim * 2, env_dim)
        self.scale = 1.0 / math.sqrt(latent_dim)

    def forward(
        self,
        z: torch.Tensor,               # (B, d)
        z_nbr: torch.Tensor,           # (B, K, d)
    ) -> torch.Tensor:
        q = self.q(z).unsqueeze(1)                         # (B, 1, d)
        k = self.k(z_nbr)                                   # (B, K, d)
        v = self.v(z_nbr)
        attn = torch.softmax((q @ k.transpose(1, 2)) * self.scale, dim=-1)  # (B,1,K)
        attn_pool = (attn @ v).squeeze(1)                   # (B, d)
        mean_pool = z_nbr.mean(dim=1)                       # (B, d)
        return self.proj(torch.cat([attn_pool, mean_pool], dim=-1))
