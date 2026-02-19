#!/usr/bin/env python3
"""Decoder for CVAE PCB Warpage.

Architecture:
  z_fused → MLP → reshape (256, 8, 8)
  → FiLMBlock(256→128, c)   8  → 16
  → FiLMBlock(128→64,  c)  16  → 32
  → FiLMBlock(64→32,   c)  32  → 64
  → FiLMBlock(32→16,   c)  64  → 128
  → Upsample + Conv(16→1) + Sigmoid   128 → 256

FiLM conditioning at every upsample block keeps the design condition active
throughout the decoding process.
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
    """Transposed-CNN decoder conditioned via FiLM at every upsampling block.

    Args:
        config      : dict from load_config() — reads c_dim.
        z_fused_dim : dimensionality of the fused latent code input
                      (64 for film/cross_attention, 128 for concat).
    """

    def __init__(self, config: dict, z_fused_dim: int):
        super().__init__()
        c_dim = int(config.get('c_dim', 64))

        # Project fused latent to initial feature map
        self.fc = nn.Linear(z_fused_dim, 256 * 8 * 8)

        # Upsampling blocks (each doubles spatial resolution)
        self.block1 = FiLMBlock(256, 128, c_dim)   # 8  → 16
        self.block2 = FiLMBlock(128,  64, c_dim)   # 16 → 32
        self.block3 = FiLMBlock( 64,  32, c_dim)   # 32 → 64
        self.block4 = FiLMBlock( 32,  16, c_dim)   # 64 → 128

        # Final upsampling to 256 — no FiLM, plain spatial output
        self.final_up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, z_fused: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_fused : (B, z_fused_dim)
            c       : (B, c_dim)  — condition for FiLM blocks

        Returns:
            x_recon : (B, 1, 256, 256)  values in [0, 1]
        """
        x = self.fc(z_fused)
        x = x.view(-1, 256, 8, 8)          # (B, 256, 8, 8)

        x = self.block1(x, c)               # (B, 128, 16, 16)
        x = self.block2(x, c)               # (B,  64, 32, 32)
        x = self.block3(x, c)               # (B,  32, 64, 64)
        x = self.block4(x, c)               # (B,  16, 128,128)

        x = self.final_up(x)                # (B,  16, 256,256)
        x = torch.sigmoid(self.final_conv(x))  # (B, 1, 256,256)
        return x
