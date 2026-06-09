#!/usr/bin/env python3
"""Enhanced Condition Encoder for DF²M.

Produces TWO condition representations:
    c_global:  (B, c_dim)          — global condition vector
    c_spatial: (B, spatial_ch, 8, 8) — spatial condition map preserving WHERE info

Compared to the original DesignEncoder:
    - c_dim = 64 (was 4)  → 16× more conditioning capacity
    - c_spatial preserved  → decoder and velocity net can use spatial information
    - Stronger CNN backbone with more capacity
"""

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class ConditionEncoder(nn.Module):
    """Enhanced design condition encoder with spatial output.

    Architecture:
        CNN branch:  design (1ch) → 4×ConvBlock → c_cnn (global) + c_spatial (8×8)
        Hand branch: selected_features → MLP → c_hand
        Fusion:      [c_cnn, c_hand] → MLP → c_global

    Args:
        config: dict with c_dim, c_cnn_dim, c_hand_dim, selected_features.
    """

    def __init__(self, config: dict):
        super().__init__()
        c_dim = int(config.get('c_dim', 64))
        c_cnn_dim = int(config.get('c_cnn_dim', 32))
        c_hand_dim = int(config.get('c_hand_dim', 32))

        selected = config.get('selected_features', list(range(24)))
        n_feat = len(selected) if isinstance(selected, (list, tuple)) else 24

        # Spatial channels at the deepest feature map (8×8 for 128×128 input)
        self.spatial_ch = 128

        # --- CNN branch ---
        # 128→64→32→16→8 spatial resolution
        self.cnn = nn.Sequential(
            _conv_block(1, 32, stride=2),     # → (32, 64, 64)
            _conv_block(32, 64, stride=2),    # → (64, 32, 32)
            _conv_block(64, 128, stride=2),   # → (128, 16, 16)
            _conv_block(128, self.spatial_ch, stride=2),  # → (128, 8, 8)
        )
        # Global vector from spatial features
        self.cnn_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.spatial_ch, c_cnn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # --- Handcrafted feature branch ---
        hand_hidden = max(n_feat * 2, 32)
        self.hand_mlp = nn.Sequential(
            nn.Linear(n_feat, hand_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hand_hidden, c_hand_dim),
            nn.ReLU(inplace=True),
        )

        # --- Fusion ---
        self.fusion = nn.Sequential(
            nn.Linear(c_cnn_dim + c_hand_dim, c_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(c_dim, c_dim),
        )

        self.c_dim = c_dim

    def forward(
        self,
        design: torch.Tensor,
        hand_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            design:        (B, 1, H, W) design image
            hand_features: (B, n_feat) selected handcrafted features

        Returns:
            c_global:  (B, c_dim)  — global condition vector
            c_spatial: (B, 128, 8, 8) — spatial condition map
        """
        # CNN branch
        c_spatial = self.cnn(design)           # (B, 128, 8, 8)
        c_cnn = self.cnn_pool(c_spatial)       # (B, c_cnn_dim)

        # Handcrafted branch
        c_hand = self.hand_mlp(hand_features)  # (B, c_hand_dim)

        # Fusion
        c_global = self.fusion(torch.cat([c_cnn, c_hand], dim=1))  # (B, c_dim)

        return c_global, c_spatial
