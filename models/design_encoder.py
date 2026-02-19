#!/usr/bin/env python3
"""Design Encoder for CVAE PCB Warpage.

Architecture:
  CNN branch  : Conv(32) → Conv(64) → Conv(128) → GlobalAvgPool → MLP → c_cnn (64)
  Hand branch : handcrafted_features (10) → MLP → c_hand (16)
  Fusion      : MLP(concat(c_cnn, c_hand)) → c (64)   [Deterministic]

Kept intentionally small to avoid memorising a handful of training designs.
Strong regularisation: Dropout + Weight Decay (applied in optimizer).
"""

import torch
import torch.nn as nn
from utils.handcrafted_features import HAND_FEATURE_DIM


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """3×3 stride-2 conv + BN + ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DesignEncoder(nn.Module):
    """Deterministic encoder that maps a design image to condition vector c.

    Args:
        config: dict from load_config() — reads c_cnn_dim, c_hand_dim, c_dim.
    """

    def __init__(self, config: dict):
        super().__init__()
        c_cnn_dim  = int(config.get('c_cnn_dim',  64))
        c_hand_dim = int(config.get('c_hand_dim', 16))
        c_dim      = int(config.get('c_dim',       64))
        dropout_p  = 0.3

        # ----- CNN branch: 256 → 128 → 64 → 32 → GlobalAvgPool → 128 -----
        self.cnn = nn.Sequential(
            _conv_block(1,   32),   # (1,256,256) → (32,128,128)
            _conv_block(32,  64),   # (32,128,128) → (64,64,64)
            _conv_block(64, 128),   # (64,64,64)   → (128,32,32)
            nn.AdaptiveAvgPool2d(1), # → (128,1,1)
        )
        self.cnn_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, c_cnn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
        )

        # ----- Handcrafted branch -----
        self.hand_mlp = nn.Sequential(
            nn.Linear(HAND_FEATURE_DIM, c_hand_dim),
            nn.ReLU(inplace=True),
        )

        # ----- Fusion MLP -----
        self.fusion = nn.Sequential(
            nn.Linear(c_cnn_dim + c_hand_dim, c_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(c_dim, c_dim),
        )

    def forward(self, design: torch.Tensor, hand_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            design       : (B, 1, H, W)  — grayscale design image in [0, 1]
            hand_features: (B, HAND_FEATURE_DIM)

        Returns:
            c : (B, c_dim)  — deterministic condition vector
        """
        c_cnn  = self.cnn_mlp(self.cnn(design))      # (B, c_cnn_dim)
        c_hand = self.hand_mlp(hand_features)          # (B, c_hand_dim)
        c      = self.fusion(torch.cat([c_cnn, c_hand], dim=1))  # (B, c_dim)
        return c
