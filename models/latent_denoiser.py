#!/usr/bin/env python3
"""Shared MLP-based denoiser / velocity network for latent-space generative models.

Used by both:
  - LDM (Latent Diffusion Model) — as a noise predictor  ε_θ(z_t, t, c)
  - LFM (Latent Flow Matching)   — as a velocity predictor v_θ(z_t, t, c)

Architecture:
  t (scalar) → SinusoidalEmbedding → MLP → t_emb (hidden_dim)
  [z_t, c] → Linear → h (hidden_dim)
  h → AdaLN-ResBlock × n_blocks → LayerNorm → Linear → output (z_dim)

Each ResBlock uses Adaptive Layer Normalisation (AdaLN) conditioned on the
timestep embedding, following the DiT (Diffusion Transformer) convention.
"""

import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for scalar timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) float — timestep values (integers for LDM, continuous [0,1] for LFM)
        Returns:
            emb: (B, dim)
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device) / half
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


class AdaLNResBlock(nn.Module):
    """Residual block with Adaptive Layer Normalisation (AdaLN).

    AdaLN(x, cond) = LN(x) * (1 + scale) + shift
    where (scale, shift) = Linear(cond)
    """

    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

        # AdaLN modulation: cond → (scale1, shift1, scale2, shift2)
        self.adaLN = nn.Linear(cond_dim, 4 * hidden_dim)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : (B, hidden_dim)
            cond : (B, cond_dim) — timestep embedding
        Returns:
            (B, hidden_dim)
        """
        params = self.adaLN(cond)  # (B, 4*hidden_dim)
        s1, b1, s2, b2 = params.chunk(4, dim=-1)

        # Block 1: AdaLN → SiLU → Linear → Dropout
        h = self.norm1(x) * (1 + s1) + b1
        h = self.drop(self.fc1(torch.nn.functional.silu(h)))

        # Block 2: AdaLN → SiLU → Linear → Dropout
        h = self.norm2(h) * (1 + s2) + b2
        h = self.drop(self.fc2(torch.nn.functional.silu(h)))

        return x + h


class LatentDenoiser(nn.Module):
    """MLP-based network for predictions in a low-dimensional latent space.

    Predicts either:
      - noise ε given (z_t, t, c)   [for LDM]
      - velocity v given (z_t, t, c) [for LFM]

    Args:
        z_dim      : latent dimension (default 64)
        c_dim      : condition vector dimension (default 64)
        t_dim      : sinusoidal embedding dimension (default 128)
        hidden_dim : MLP hidden width (default 512)
        n_blocks   : number of AdaLN residual blocks (default 8)
        dropout    : dropout rate (default 0.1)
    """

    def __init__(
        self,
        z_dim: int = 64,
        c_dim: int = 64,
        t_dim: int = 128,
        hidden_dim: int = 512,
        n_blocks: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.z_dim = z_dim

        # Timestep embedding: scalar → t_dim → hidden_dim
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(t_dim),
            nn.Linear(t_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection: [z_t, c] → hidden_dim
        self.input_proj = nn.Linear(z_dim + c_dim, hidden_dim)

        # Residual blocks with AdaLN (conditioned on t_emb)
        self.blocks = nn.ModuleList([
            AdaLNResBlock(hidden_dim, hidden_dim, dropout)
            for _ in range(n_blocks)
        ])

        # Output projection
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, z_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t : (B, z_dim) — noisy / interpolated latent
            t   : (B,) — timestep (integer for LDM, float in [0,1] for LFM)
            c   : (B, c_dim) — condition vector from DesignEncoder

        Returns:
            prediction : (B, z_dim) — predicted noise (LDM) or velocity (LFM)
        """
        t_emb = self.time_embed(t.float())  # (B, hidden_dim)

        h = self.input_proj(torch.cat([z_t, c], dim=-1))  # (B, hidden_dim)

        for block in self.blocks:
            h = block(h, t_emb)

        return self.out_proj(self.out_norm(h))  # (B, z_dim)
