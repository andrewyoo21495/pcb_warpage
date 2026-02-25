#!/usr/bin/env python3
"""U-Net noise predictor for Conditional DDPM.

Architecture:
  - Sinusoidal timestep embedding + MLP -> t_emb
  - Combined conditioning: t_emb + global_cond (element-wise sum, dim=cond_dim)
  - AdaGN (Adaptive Group Normalization) in every ResBlock
  - Spatial design features concatenated at matching U-Net resolutions

Channel schedule: [64, 128, 256, 256] at 256->128->64->32, bottleneck at 16x16.

Spatial concat points:
  - 128x128: concat feat_128 (64ch) with U-Net (64ch)  -> 128 ch
  - 64x64:   concat feat_64 (128ch) with U-Net (128ch) -> 256 ch
  - 32x32:   concat feat_32 (256ch) with U-Net (256ch) -> 512 -> project to 256
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ddpm_condition_encoder import FEAT_CH_128, FEAT_CH_64, FEAT_CH_32


# ======================================================================
# Building blocks
# ======================================================================

class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timesteps -> MLP."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer timesteps
        Returns:
            (B, out_dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class AdaGN(nn.Module):
    """Adaptive Group Normalization: GN + learned scale/shift from condition."""

    def __init__(self, num_channels: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)
        self.proj = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : (B, C, H, W)
            cond : (B, cond_dim)
        """
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        x = self.gn(x)
        return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResBlock(nn.Module):
    """ResBlock with AdaGN conditioning and optional dropout.

    norm1 -> SiLU -> conv1 -> norm2 -> SiLU -> dropout -> conv2 + skip
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = AdaGN(in_ch, cond_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = AdaGN(out_ch, cond_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, cond)))
        h = self.dropout(h)
        h = self.conv2(F.silu(self.norm2(h, cond)))
        return h + self.skip(x)


class Downsample(nn.Module):
    """2x spatial downsampling via strided convolution."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2x spatial upsampling via interpolation + convolution."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ======================================================================
# U-Net
# ======================================================================

class UNet(nn.Module):
    """U-Net noise predictor with AdaGN and spatial condition injection.

    Args:
        config: dict from load_config().
    """

    def __init__(self, config: dict):
        super().__init__()
        base_ch  = int(config.get('ddpm_base_ch', 64))
        cond_dim = int(config.get('ddpm_cond_dim', 256))
        dropout  = float(config.get('ddpm_dropout', 0.15))

        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 4]  # [64, 128, 256, 256]
        t_emb_dim = 128

        # --- Timestep embedding ---
        self.time_embed = SinusoidalTimestepEmbedding(t_emb_dim, cond_dim)

        # --- Initial conv ---
        self.init_conv = nn.Conv2d(1, ch[0], 3, padding=1)   # 256x256

        # --- Encoder ---
        # Level 0: 256x256, ch[0]=64
        self.enc_res0 = ResBlock(ch[0], ch[0], cond_dim, dropout)
        self.down0 = Downsample(ch[0])

        # Level 1: 128x128, concat feat_128 (64ch) -> ch[0]+FEAT_CH_128=128
        self.enc_proj1 = nn.Conv2d(ch[0] + FEAT_CH_128, ch[1], 1)
        self.enc_res1 = ResBlock(ch[1], ch[1], cond_dim, dropout)
        self.down1 = Downsample(ch[1])

        # Level 2: 64x64, concat feat_64 (128ch) -> ch[1]+FEAT_CH_64=256
        self.enc_proj2 = nn.Conv2d(ch[1] + FEAT_CH_64, ch[2], 1)
        self.enc_res2 = ResBlock(ch[2], ch[2], cond_dim, dropout)
        self.down2 = Downsample(ch[2])

        # Level 3: 32x32, concat feat_32 (256ch) -> ch[2]+FEAT_CH_32=512
        self.enc_proj3 = nn.Conv2d(ch[2] + FEAT_CH_32, ch[3], 1)
        self.enc_res3 = ResBlock(ch[3], ch[3], cond_dim, dropout)
        self.down3 = Downsample(ch[3])

        # --- Bottleneck: 16x16 ---
        self.bot_res1 = ResBlock(ch[3], ch[3], cond_dim, dropout)
        self.bot_res2 = ResBlock(ch[3], ch[3], cond_dim, dropout)

        # --- Decoder ---
        self.up3 = Upsample(ch[3])
        self.dec_proj3 = nn.Conv2d(ch[3] + ch[3], ch[3], 1)  # skip concat
        self.dec_res3 = ResBlock(ch[3], ch[3], cond_dim, dropout)

        self.up2 = Upsample(ch[3])
        self.dec_proj2 = nn.Conv2d(ch[3] + ch[2], ch[2], 1)
        self.dec_res2 = ResBlock(ch[2], ch[2], cond_dim, dropout)

        self.up1 = Upsample(ch[2])
        self.dec_proj1 = nn.Conv2d(ch[2] + ch[1], ch[1], 1)
        self.dec_res1 = ResBlock(ch[1], ch[1], cond_dim, dropout)

        self.up0 = Upsample(ch[1])
        self.dec_proj0 = nn.Conv2d(ch[1] + ch[0], ch[0], 1)
        self.dec_res0 = ResBlock(ch[0], ch[0], cond_dim, dropout)

        # --- Output ---
        self.final_norm = nn.GroupNorm(8, ch[0])
        self.final_conv = nn.Conv2d(ch[0], 1, 3, padding=1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        global_cond: torch.Tensor,
        spatial_feats: list[torch.Tensor],
    ) -> torch.Tensor:
        """Predict noise from noisy input.

        Args:
            x_t           : (B, 1, 256, 256) noisy elevation
            t             : (B,) integer timesteps
            global_cond   : (B, cond_dim) from condition encoder
            spatial_feats : [feat_128, feat_64, feat_32]

        Returns:
            noise_pred : (B, 1, 256, 256)
        """
        feat_128, feat_64, feat_32 = spatial_feats

        # Combined conditioning: timestep + global design condition
        cond = self.time_embed(t) + global_cond   # (B, cond_dim)

        # --- Encoder ---
        h = self.init_conv(x_t)                    # (B, 64, 256, 256)
        h0 = self.enc_res0(h, cond)                # skip_0: (B, 64, 256)
        h = self.down0(h0)                         # (B, 64, 128, 128)

        h = self.enc_proj1(torch.cat([h, feat_128], dim=1))  # -> (B, 128, 128)
        h1 = self.enc_res1(h, cond)                # skip_1: (B, 128, 128)
        h = self.down1(h1)                         # (B, 128, 64, 64)

        h = self.enc_proj2(torch.cat([h, feat_64], dim=1))   # -> (B, 256, 64)
        h2 = self.enc_res2(h, cond)                # skip_2: (B, 256, 64)
        h = self.down2(h2)                         # (B, 256, 32, 32)

        h = self.enc_proj3(torch.cat([h, feat_32], dim=1))   # -> (B, 256, 32)
        h3 = self.enc_res3(h, cond)                # skip_3: (B, 256, 32)
        h = self.down3(h3)                         # (B, 256, 16, 16)

        # --- Bottleneck ---
        h = self.bot_res1(h, cond)
        h = self.bot_res2(h, cond)

        # --- Decoder ---
        h = self.up3(h)                            # (B, 256, 32)
        h = self.dec_proj3(torch.cat([h, h3], dim=1))
        h = self.dec_res3(h, cond)

        h = self.up2(h)                            # (B, 256, 64)
        h = self.dec_proj2(torch.cat([h, h2], dim=1))
        h = self.dec_res2(h, cond)

        h = self.up1(h)                            # (B, 256, 128)
        h = self.dec_proj1(torch.cat([h, h1], dim=1))
        h = self.dec_res1(h, cond)

        h = self.up0(h)                            # (B, 128, 256)
        h = self.dec_proj0(torch.cat([h, h0], dim=1))
        h = self.dec_res0(h, cond)

        # --- Output ---
        h = F.silu(self.final_norm(h))
        return self.final_conv(h)                  # (B, 1, 256, 256)
