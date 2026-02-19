#!/usr/bin/env python3
"""Elevation Encoder for CVAE PCB Warpage.

Architecture:
  Conv(32) → Conv(64) → Conv(128) → Conv(256) → AdaptiveAvgPool(4,4)
  → Flatten → MLP(4096→512)
  → mu (z_dim),  logvar (z_dim)
  → z1 = mu + eps * exp(0.5 * logvar)   [Reparameterisation]

Larger capacity than the design encoder; z1 quality determines sample diversity.
All stochasticity is concentrated here — design encoder remains deterministic.
"""

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """3×3 stride-2 conv + BN + LeakyReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class ElevationEncoder(nn.Module):
    """Stochastic encoder that maps an elevation image to (mu, logvar, z1).

    Args:
        config: dict from load_config() — reads z_dim.
    """

    def __init__(self, config: dict):
        super().__init__()
        z_dim = int(config.get('z_dim', 64))

        # Spatial: 256 → 128 → 64 → 32 → 16  (4× stride-2 convs)
        self.cnn = nn.Sequential(
            _conv_block(1,   32),   # (1,256,256)  → (32,128,128)
            _conv_block(32,  64),   # (32,128,128) → (64,64,64)
            _conv_block(64, 128),   # (64,64,64)   → (128,32,32)
            _conv_block(128,256),   # (128,32,32)  → (256,16,16)
            nn.AdaptiveAvgPool2d(4), # → (256,4,4)
        )
        flat_dim = 256 * 4 * 4  # 4096

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mu_layer     = nn.Linear(512, z_dim)
        self.logvar_layer = nn.Linear(512, z_dim)

    def forward(self, elevation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            elevation: (B, 1, H, W)  — normalised elevation image in [0, 1]

        Returns:
            mu     : (B, z_dim)
            logvar : (B, z_dim)
        """
        h      = self.mlp(self.cnn(elevation))
        mu     = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z1 via the reparameterisation trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
