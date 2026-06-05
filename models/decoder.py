#!/usr/bin/env python3
"""Decoder for CVAE PCB Warpage.

Architecture:
  z_fused → MLP → reshape (256, 8, 8)
  → PlainBlock(256→128)       8  → 16   (z1-driven, no condition)
  → PlainBlock(128→64)       16  → 32   (z1-driven, no condition)
  → FiLMBlock(64→32,   c)   32  → 64   (fine detail, c-conditioned)
  → FiLMBlock(32→16,   c)   64  → 128  (fine detail, c-conditioned)
  → Upsample + Conv(16→1) + Sigmoid   128 → 256

Coarse layers (block1-2) rely solely on z1 so the latent code must carry
the overall elevation structure.  FiLM conditioning is applied only at
fine-resolution layers (block3-4) for design-specific detail adjustment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: x = gamma(c) * x + beta(c)."""

    def __init__(self, num_channels: int, c_dim: int):
        super().__init__()
        self.gamma = nn.Linear(c_dim, num_channels)
        self.beta  = nn.Linear(c_dim, num_channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W)
            c : (B, c_dim)
        Returns:
            modulated feature map (B, C, H, W)
        """
        gamma = self.gamma(c).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta  = self.beta(c).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        return gamma * x + beta


class PlainBlock(nn.Module):
    """Upsample → Conv → BN → ReLU  (no condition dependency)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        return x


class FiLMBlock(nn.Module):
    """Upsample → Conv → BN → ReLU → FiLM."""

    def __init__(self, in_ch: int, out_ch: int, c_dim: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.film = FiLMLayer(out_ch, c_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.film(x, c)
        return x


class Decoder(nn.Module):
    """Transposed-CNN decoder with FiLM conditioning only at fine-resolution layers.

    Coarse layers (block1-2) use no condition — forcing z1 to carry the
    overall elevation structure.  Fine layers (block3-4) apply FiLM so
    the design condition can adjust spatial details.

    Args:
        config      : dict from load_config() — reads c_dim.
        z_fused_dim : dimensionality of the fused latent code input
                      (z_dim for film/cross_attention, z_dim+c_dim for concat).
    """

    def __init__(self, config: dict, z_fused_dim: int):
        super().__init__()
        c_dim = int(config.get('c_dim', 64))

        # Project fused latent to initial feature map
        self.fc = nn.Linear(z_fused_dim, 256 * 8 * 8)

        # --- Coarse layers: no FiLM (z1-driven) ---
        self.block1 = PlainBlock(256, 128)   # 8  → 16
        self.block2 = PlainBlock(128,  64)   # 16 → 32

        # --- Fine layers: FiLM-conditioned ---
        self.block3 = FiLMBlock( 64,  32, c_dim)   # 32 → 64
        self.block4 = FiLMBlock( 32,  16, c_dim)   # 64 → 128

        # Final upsampling to 256 — no FiLM, plain spatial output
        self.final_up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, z_fused: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_fused : (B, z_fused_dim)
            c       : (B, c_dim)  — condition for FiLM blocks (block3-4 only)

        Returns:
            x_recon : (B, 1, 256, 256)  values in [0, 1]
        """
        x = self.fc(z_fused)
        x = x.view(-1, 256, 8, 8)          # (B, 256, 8, 8)

        x = self.block1(x)                  # (B, 128, 16, 16)
        x = self.block2(x)                  # (B,  64, 32, 32)
        x = self.block3(x, c)               # (B,  32, 64, 64)
        x = self.block4(x, c)               # (B,  16, 128,128)

        x = self.final_up(x)                # (B,  16, 256,256)
        x = torch.sigmoid(self.final_conv(x))  # (B, 1, 256,256)
        return x
