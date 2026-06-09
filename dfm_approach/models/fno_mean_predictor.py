#!/usr/bin/env python3
"""Module A: Fourier Neural Operator (FNO) Mean Predictor.

Learns the deterministic mapping:  design_image + features → mean_warpage.

Architecture:
    Lifting (1→width) → N×[SpectralConv2d + Conv1x1 + FiLM] → Projection (width→1)

SpectralConv operates in the frequency domain:
    FFT → learnable mode filter → IFFT

FiLM conditioning injects handcrafted features at each Fourier layer:
    γ(feat) · x + β(feat)

References:
    Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations",
    NeurIPS 2021 — arXiv:2010.08895
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Spectral Convolution
# ------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    """2D Spectral Convolution via FFT.

    Learns a complex-valued weight tensor that multiplies the top-k
    frequency modes, then inverts back to spatial domain.

    Args:
        in_channels:  input channels
        out_channels: output channels
        modes1:       number of Fourier modes to keep along height axis
        modes2:       number of Fourier modes to keep along width axis
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def _compl_mul2d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication: (B, C_in, H, W) × (C_in, C_out, H, W) → (B, C_out, H, W)."""
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # FFT along spatial dims
        x_ft = torch.fft.rfft2(x.float(), norm='ortho')

        H_ft, W_ft = x_ft.shape[-2], x_ft.shape[-1]
        m1 = min(self.modes1, H_ft)
        m2 = min(self.modes2, W_ft)

        out_ft = torch.zeros(B, self.out_channels, H_ft, W_ft,
                             dtype=torch.cfloat, device=x.device)

        # Top-left corner modes
        out_ft[:, :, :m1, :m2] = self._compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        # Bottom-left corner modes (negative frequencies along height)
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )

        # Inverse FFT
        return torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]), norm='ortho')


# ------------------------------------------------------------------
# FiLM conditioning layer
# ------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: γ(feat) · x + β(feat)."""

    def __init__(self, feat_dim: int, channels: int):
        super().__init__()
        self.gamma = nn.Linear(feat_dim, channels)
        self.beta = nn.Linear(feat_dim, channels)

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, C, H, W)
            feat: (B, feat_dim)
        """
        gamma = self.gamma(feat).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta(feat).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


# ------------------------------------------------------------------
# FNO Fourier Layer (one block)
# ------------------------------------------------------------------

class FourierLayer(nn.Module):
    """Single Fourier layer: SpectralConv + Conv1x1 + FiLM + activation."""

    def __init__(self, width: int, modes: int, feat_dim: int):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes, modes)
        self.conv = nn.Conv2d(width, width, kernel_size=1)
        self.film = FiLMLayer(feat_dim, width)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        x1 = self.spectral_conv(x)
        x2 = self.conv(x)
        x = x1 + x2
        x = self.film(x, feat)
        x = self.norm(x)
        x = F.gelu(x)
        return x


# ------------------------------------------------------------------
# FNO Mean Predictor
# ------------------------------------------------------------------

class FNOMeanPredictor(nn.Module):
    """Fourier Neural Operator for mean warpage prediction.

    Architecture:
        design_image (1ch) → Lift (1→width) → N Fourier layers → Project (width→1)
        Handcrafted features injected via FiLM at each Fourier layer.

    Args:
        config: dict with keys fno_width, fno_modes, fno_num_layers, fno_feat_dim,
                selected_features (list of ints).
    """

    def __init__(self, config: dict):
        super().__init__()
        width = int(config.get('fno_width', 32))
        modes = int(config.get('fno_modes', 16))
        n_layers = int(config.get('fno_num_layers', 4))

        # Feature input dimension
        selected = config.get('selected_features', list(range(24)))
        if isinstance(selected, (list, tuple)):
            n_feat = len(selected)
        else:
            n_feat = 24
        feat_embed_dim = int(config.get('fno_feat_dim', 64))

        # Feature embedding MLP
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, feat_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_embed_dim, feat_embed_dim),
            nn.ReLU(inplace=True),
        )

        # Lifting: 1ch → width
        self.lift = nn.Conv2d(1, width, kernel_size=1)

        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer(width, modes, feat_embed_dim) for _ in range(n_layers)
        ])

        # Projection: width → 1ch
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, 1, kernel_size=1),
        )

    def forward(
        self,
        design: torch.Tensor,
        hand_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            design:        (B, 1, H, W) design image
            hand_features: (B, n_feat) selected handcrafted features

        Returns:
            mean_warpage:  (B, 1, H, W) predicted mean warpage in [0, 1]
        """
        feat = self.feat_mlp(hand_features)  # (B, feat_embed_dim)

        x = self.lift(design)  # (B, width, H, W)

        for layer in self.fourier_layers:
            x = layer(x, feat)

        x = self.project(x)  # (B, 1, H, W)
        return torch.sigmoid(x)

    @torch.no_grad()
    def predict(
        self,
        design: torch.Tensor,
        hand_features: torch.Tensor,
    ) -> torch.Tensor:
        """Inference-mode forward pass."""
        return self.forward(design, hand_features)
