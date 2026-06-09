#!/usr/bin/env python3
"""Module B-1: Residual Conditional Autoencoder (CAE).

Learns the latent representation of residual warpage patterns:
    residual (ε) = elevation - mean_warpage

Components:
    ResidualEncoder:  ε → (μ_z, log_var) → z ~ N(μ_z, σ²)
    ResidualDecoder:  z + c_global + c_spatial → ε̂ (reconstructed residual)

Key improvements over the original CVAE:
    1. Operates on RESIDUALS (lower amplitude, simpler structure)
    2. c_dim=64 (was 4) — prevents condition-ignoring
    3. c_spatial (8×8) preserved for multi-scale conditioning
    4. Soft free bits for KL collapse prevention
    5. Tanh output (residuals can be negative)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Residual Encoder
# ------------------------------------------------------------------

class ResidualEncoder(nn.Module):
    """Encodes residual images to a stochastic latent z.

    Architecture:
        residual (1, H, W) → 4× ConvBlock → AdaptiveAvgPool → FC → (μ, log_var)
    """

    def __init__(self, z_dim: int = 64, logvar_clamp: tuple[float, float] = (-6.0, 4.0)):
        super().__init__()
        self.z_dim = z_dim
        self.logvar_min, self.logvar_max = logvar_clamp

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),    nn.BatchNorm2d(32),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),   nn.BatchNorm2d(64),  nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),  nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            residual: (B, 1, H, W) residual image

        Returns:
            mu:     (B, z_dim)
            logvar: (B, z_dim)
        """
        h = self.encoder(residual)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar


# ------------------------------------------------------------------
# Spatial Adaptation block
# ------------------------------------------------------------------

class SpatialAdapt(nn.Module):
    """Injects spatial condition information into decoder features.

    Downsamples c_spatial to match the current feature map size, then
    applies learned channel mixing.
    """

    def __init__(self, spatial_ch: int, out_ch: int):
        super().__init__()
        self.adapt = nn.Sequential(
            nn.Conv2d(spatial_ch, out_ch, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, c_spatial: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:         (B, C, H, W) current feature map
            c_spatial: (B, spatial_ch, 8, 8) spatial condition
        """
        c = F.interpolate(c_spatial, size=x.shape[-2:], mode='bilinear', align_corners=False)
        c = self.adapt(c)
        return x + c


# ------------------------------------------------------------------
# FiLM block for decoder
# ------------------------------------------------------------------

class DecoderFiLM(nn.Module):
    """FiLM modulation from global condition for decoder blocks."""

    def __init__(self, c_dim: int, channels: int):
        super().__init__()
        self.gamma = nn.Linear(c_dim, channels)
        self.beta = nn.Linear(c_dim, channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        g = self.gamma(c).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(c).unsqueeze(-1).unsqueeze(-1)
        return g * x + b


# ------------------------------------------------------------------
# Residual Decoder
# ------------------------------------------------------------------

class ResidualDecoder(nn.Module):
    """Decodes latent z to residual image with multi-scale conditioning.

    Block 1 (8×8):  ConvTranspose + SpatialAdapt(c_spatial at 8×8)
    Block 2 (16×16): ConvTranspose + FiLM(c_global)
    Block 3 (32×32): ConvTranspose + FiLM(c_global)
    Block 4 (64→128): ConvTranspose → Tanh
    """

    def __init__(self, z_dim: int = 64, c_dim: int = 64, spatial_ch: int = 128):
        super().__init__()
        self.z_dim = z_dim

        # z_fused → initial feature map
        self.fc = nn.Linear(z_dim, 256 * 8 * 8)

        # Block 1: (256, 8, 8) → (128, 16, 16) + spatial conditioning
        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.spatial_adapt = SpatialAdapt(spatial_ch, 128)

        # Block 2: (128, 16, 16) → (64, 32, 32) + FiLM
        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.film2 = DecoderFiLM(c_dim, 64)

        # Block 3: (64, 32, 32) → (32, 64, 64) + FiLM
        self.up3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.film3 = DecoderFiLM(c_dim, 32)

        # Block 4: (32, 64, 64) → (1, 128, 128) → Tanh
        self.up4 = nn.ConvTranspose2d(32, 1, 4, 2, 1)

    def forward(
        self,
        z: torch.Tensor,
        c_global: torch.Tensor,
        c_spatial: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z:         (B, z_dim) latent vector
            c_global:  (B, c_dim) global condition
            c_spatial: (B, spatial_ch, 8, 8) spatial condition

        Returns:
            residual: (B, 1, H, W) reconstructed residual (range ~[-1, 1])
        """
        x = self.fc(z).view(-1, 256, 8, 8)

        # Block 1 + spatial conditioning at native 8×8 resolution
        x = F.leaky_relu(self.bn1(self.up1(x)), 0.2)   # (B, 128, 16, 16)
        x = self.spatial_adapt(x, c_spatial)

        # Block 2 + FiLM
        x = F.leaky_relu(self.bn2(self.up2(x)), 0.2)   # (B, 64, 32, 32)
        x = self.film2(x, c_global)

        # Block 3 + FiLM
        x = F.leaky_relu(self.bn3(self.up3(x)), 0.2)   # (B, 32, 64, 64)
        x = self.film3(x, c_global)

        # Block 4 → output
        x = torch.tanh(self.up4(x))                      # (B, 1, 128, 128)
        return x


# ------------------------------------------------------------------
# Combined Residual CAE
# ------------------------------------------------------------------

class ResidualCAE(nn.Module):
    """Residual Conditional Autoencoder.

    Combines ResidualEncoder + ResidualDecoder with FiLM-based
    latent-condition fusion.

    Training:  forward(residual, c_global, c_spatial) → ε̂, μ, logvar
    Inference: decode(z, c_global, c_spatial) → ε̂
    """

    def __init__(self, config: dict):
        super().__init__()
        z_dim = int(config.get('z_dim', 64))
        c_dim = int(config.get('c_dim', 64))
        spatial_ch = 128  # matches ConditionEncoder output
        logvar_clamp = (
            float(config.get('cae_logvar_clamp_min', -6.0)),
            float(config.get('cae_logvar_clamp_max', 4.0)),
        )

        self.encoder = ResidualEncoder(z_dim, logvar_clamp)
        self.decoder = ResidualDecoder(z_dim, c_dim, spatial_ch)

        # FiLM fusion: c_global modulates z before decoding
        self.film_gamma = nn.Linear(c_dim, z_dim)
        self.film_beta = nn.Linear(c_dim, z_dim)

        self.z_dim = z_dim
        self.c_dim = c_dim

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _fuse(self, z: torch.Tensor, c_global: torch.Tensor) -> torch.Tensor:
        gamma = self.film_gamma(c_global)
        beta = self.film_beta(c_global)
        return gamma * z + beta

    def forward(
        self,
        residual: torch.Tensor,
        c_global: torch.Tensor,
        c_spatial: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass.

        Args:
            residual:  (B, 1, H, W) residual image
            c_global:  (B, c_dim)
            c_spatial: (B, 128, 8, 8)

        Returns:
            recon:  (B, 1, H, W) reconstructed residual
            mu:     (B, z_dim)
            logvar: (B, z_dim)
        """
        mu, logvar = self.encoder(residual)
        z = self._reparameterize(mu, logvar)
        z_fused = self._fuse(z, c_global)
        recon = self.decoder(z_fused, c_global, c_spatial)
        return recon, mu, logvar

    def encode(
        self,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode residual to latent parameters (for OT-CFM training)."""
        return self.encoder(residual)

    def decode(
        self,
        z: torch.Tensor,
        c_global: torch.Tensor,
        c_spatial: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent z to residual image (for OT-CFM inference)."""
        z_fused = self._fuse(z, c_global)
        return self.decoder(z_fused, c_global, c_spatial)
