#!/usr/bin/env python3
"""Multi-scale CNN condition encoder for Conditional DDPM.

Outputs:
  - Spatial feature maps at 128x128, 64x64, 32x32 (for U-Net concat)
  - Global condition vector (B, cond_dim) (for AdaGN injection)

Architecture:
    stem   Conv(1->32, stride=1)    256x256
    down1  Conv(32->64, stride=2)   128x128  -> feat_128
    down2  Conv(64->128, stride=2)   64x64   -> feat_64
    down3  Conv(128->256, stride=2)  32x32   -> feat_32
    global AdaptiveAvgPool + hand_features MLP -> (B, cond_dim)
"""

import torch
import torch.nn as nn

from utils.handcrafted_features import HAND_FEATURE_DIM

# Channel counts at each spatial scale (referenced by unet.py)
FEAT_CH_128 = 64
FEAT_CH_64 = 128
FEAT_CH_32 = 256


class DDPMConditionEncoder(nn.Module):
    """Multi-scale design condition encoder for DDPM.

    Args:
        config: dict from load_config().
    """

    def __init__(self, config: dict):
        super().__init__()
        cond_dim = int(config.get('ddpm_cond_dim', 256))
        hand_input_dim = int(config.get('hand_input_dim', HAND_FEATURE_DIM))

        # --- Spatial CNN ---
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, stride=1),
            nn.SiLU(),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32, FEAT_CH_128, 3, padding=1, stride=2),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(FEAT_CH_128, FEAT_CH_64, 3, padding=1, stride=2),
            nn.SiLU(),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(FEAT_CH_64, FEAT_CH_32, 3, padding=1, stride=2),
            nn.SiLU(),
        )

        # --- Global condition ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cnn_fc = nn.Sequential(
            nn.Linear(FEAT_CH_32, 256),
            nn.SiLU(),
        )
        self.hand_mlp = nn.Sequential(
            nn.Linear(hand_input_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
        )
        self.fuse_fc = nn.Sequential(
            nn.Linear(256 + 128, cond_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        design: torch.Tensor,
        hand_features: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Args:
            design        : (B, 1, H, W) float32 in [0, 1]
            hand_features : (B, hand_input_dim)

        Returns:
            spatial_feats : [feat_128, feat_64, feat_32]
            global_cond   : (B, cond_dim)
        """
        x = self.stem(design)
        feat_128 = self.down1(x)
        feat_64 = self.down2(feat_128)
        feat_32 = self.down3(feat_64)

        pooled = self.global_pool(feat_32).flatten(1)
        cnn_vec = self.cnn_fc(pooled)
        hand_vec = self.hand_mlp(hand_features)
        global_cond = self.fuse_fc(torch.cat([cnn_vec, hand_vec], dim=1))

        return [feat_128, feat_64, feat_32], global_cond
