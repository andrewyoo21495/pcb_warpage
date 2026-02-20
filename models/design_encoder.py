#!/usr/bin/env python3
"""Design Encoder for CVAE PCB Warpage.

Architecture:
  CNN branch  : Conv(32) → Conv(64) → Conv(128)
                → ChannelBottleneck(64, 1×1) → SpatialPool(4×4) → MLP → c_cnn (64)
  Hand branch : handcrafted_features (N) → MLP → c_hand (32)
  Fusion      : MLP(concat(c_cnn, c_hand)) → c (64)   [Deterministic]

The 4×4 spatial pool (instead of 1×1 GlobalAvgPool) preserves coarse spatial layout:
each of the 16 cells captures what is happening in that region of the design image,
so the CNN can distinguish designs where features are in different locations even if
the global statistics (density, orientation) are similar.

When ``selected_features`` is set in config (comma-separated indices),
only those features are fed to the hand branch (N = len(selected)).
Otherwise all 22 features are used.

Kept intentionally small to avoid memorising a handful of training designs.
Strong regularisation: Dropout + Weight Decay (applied in optimizer).
"""

import torch
import torch.nn as nn
from utils.handcrafted_features import HAND_FEATURE_DIM


def _parse_selected_features(config: dict) -> list[int] | None:
    """Parse ``selected_features`` from config.

    Returns a sorted list of integer indices, or None if the key is absent.
    """
    raw = config.get('selected_features', None)
    if raw is None:
        return None
    if isinstance(raw, list):
        indices = [int(x) for x in raw]
    else:
        indices = [int(x.strip()) for x in str(raw).split(',') if x.strip()]
    if not indices:
        return None
    for idx in indices:
        if idx < 0 or idx >= HAND_FEATURE_DIM:
            raise ValueError(
                f"selected_features index {idx} out of range [0, {HAND_FEATURE_DIM})"
            )
    return sorted(set(indices))


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
        config: dict from load_config() — reads c_cnn_dim, c_hand_dim, c_dim,
                and optionally selected_features.
    """

    def __init__(self, config: dict):
        super().__init__()
        c_cnn_dim  = int(config.get('c_cnn_dim',  64))
        c_hand_dim = int(config.get('c_hand_dim', 32))
        c_dim      = int(config.get('c_dim',       64))
        dropout_p  = 0.3

        # ----- Feature selection -----
        self.selected_indices = _parse_selected_features(config)
        if self.selected_indices is not None:
            hand_input_dim = len(self.selected_indices)
            self.register_buffer(
                '_feat_indices',
                torch.tensor(self.selected_indices, dtype=torch.long),
            )
        else:
            hand_input_dim = HAND_FEATURE_DIM

        # ----- CNN branch -----
        # Stride-2 convs reduce 256→128→64→32, then:
        #   1×1 conv: channel bottleneck 128→64 (keeps param count small)
        #   AdaptiveAvgPool2d(4): spatial pool to 4×4 (preserves layout info)
        # Flattened: 64 × 4 × 4 = 1024
        self.cnn = nn.Sequential(
            _conv_block(1,   32),                           # (1,256,256) → (32,128,128)
            _conv_block(32,  64),                           # (32,128,128) → (64,64,64)
            _conv_block(64, 128),                           # (64,64,64)   → (128,32,32)
            nn.Conv2d(128, 64, kernel_size=1, bias=False),  # channel bottleneck → (64,32,32)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),                        # spatial pool  → (64,4,4)
        )
        self.cnn_mlp = nn.Sequential(
            nn.Flatten(),                        # 64×4×4 = 1024
            nn.Linear(1024, c_cnn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
        )

        # ----- Handcrafted branch -----
        hand_hidden = max(c_hand_dim, hand_input_dim + 8)
        self.hand_mlp = nn.Sequential(
            nn.Linear(hand_input_dim, hand_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hand_hidden, c_hand_dim),
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
            hand_features: (B, HAND_FEATURE_DIM)  — always full 22 features;
                           selection is applied internally if configured.

        Returns:
            c : (B, c_dim)  — deterministic condition vector
        """
        c_cnn = self.cnn_mlp(self.cnn(design))  # (B, c_cnn_dim)

        # Apply feature selection if configured
        if self.selected_indices is not None:
            hand_features = hand_features[:, self._feat_indices]

        c_hand = self.hand_mlp(hand_features)    # (B, c_hand_dim)
        c      = self.fusion(torch.cat([c_cnn, c_hand], dim=1))  # (B, c_dim)
        return c
