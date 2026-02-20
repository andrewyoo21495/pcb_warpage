#!/usr/bin/env python3
"""Full CVAE model for PCB Warpage Distribution Generation.

Training flow:
  Design(C) → DesignEncoder → c   (deterministic)
  Elevation(X) → ElevationEncoder → (mu, logvar) → z1  (stochastic)
  z_fused = Fuse(z1, c)           (Concat / FiLM / CrossAttention)
  x_recon = Decoder(z_fused, c)   (FiLM-conditioned upsampling)

Inference flow:
  New Design → DesignEncoder → c
  z1 ~ N(0, I)  ×  K samples
  x_gen = Decoder(Fuse(z1, c), c)

Fusion strategies (config key: fusion_method):
  'concat'          : z = cat(z1, c)             z_fused_dim = z_dim + c_dim
  'film'            : z = gamma(c)*z1 + beta(c)  z_fused_dim = z_dim       [default]
  'cross_attention' : z = Attn(Q=c, K=z1, V=z1) z_fused_dim = z_dim
"""

import torch
import torch.nn as nn

from models.design_encoder   import DesignEncoder
from models.elevation_encoder import ElevationEncoder
from models.decoder           import Decoder


class CVAE(nn.Module):
    """Conditional Variational Autoencoder for PCB warpage distribution modelling.

    Args:
        config: dict from load_config().
    """

    def __init__(self, config: dict):
        super().__init__()
        self.z_dim         = int(config.get('z_dim', 64))
        self.c_dim         = int(config.get('c_dim', 64))
        self.fusion_method = str(config.get('fusion_method', 'film'))

        # Encoders
        self.design_encoder    = DesignEncoder(config)
        self.elevation_encoder = ElevationEncoder(config)

        # Fused latent dimension
        if self.fusion_method == 'concat':
            z_fused_dim = self.z_dim + self.c_dim   # 128
        else:
            z_fused_dim = self.z_dim                 # 64

        # Fusion-specific layers
        if self.fusion_method == 'film':
            self.film_gamma = nn.Linear(self.c_dim, self.z_dim)
            self.film_beta  = nn.Linear(self.c_dim, self.z_dim)

        elif self.fusion_method == 'cross_attention':
            embed_dim = self.z_dim  # 64 (must be divisible by num_heads)
            self.q_proj  = nn.Linear(self.c_dim, embed_dim)
            self.kv_proj = nn.Linear(self.z_dim, embed_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=4,
                batch_first=True,
            )

        # Decoder
        self.decoder = Decoder(config, z_fused_dim)

    # ------------------------------------------------------------------
    # Sub-operations
    # ------------------------------------------------------------------

    def encode(
        self,
        elevation: torch.Tensor,
        design: torch.Tensor,
        hand_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode inputs to (mu, logvar, c)."""
        c              = self.design_encoder(design, hand_features)
        mu, logvar     = self.elevation_encoder(elevation)
        return mu, logvar, c

    def fuse(self, z1: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Fuse stochastic latent z1 with condition c."""
        if self.fusion_method == 'concat':
            return torch.cat([z1, c], dim=1)

        elif self.fusion_method == 'film':
            gamma = self.film_gamma(c)
            beta  = self.film_beta(c)
            return gamma * z1 + beta

        elif self.fusion_method == 'cross_attention':
            Q  = self.q_proj(c).unsqueeze(1)    # (B, 1, embed_dim)
            K  = self.kv_proj(z1).unsqueeze(1)  # (B, 1, embed_dim)
            V  = K
            z, _ = self.cross_attn(Q, K, V)
            return z.squeeze(1)                  # (B, embed_dim)

        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method!r}. "
                             "Choose 'concat', 'film', or 'cross_attention'.")

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        elevation: torch.Tensor,
        design: torch.Tensor,
        hand_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full CVAE forward pass for training.

        Args:
            elevation    : (B, 1, H, W)
            design       : (B, 1, H, W)
            hand_features: (B, HAND_FEATURE_DIM)

        Returns:
            x_recon : (B, 1, H, W)  — reconstructed elevation
            mu      : (B, z_dim)
            logvar  : (B, z_dim)
        """
        mu, logvar, c = self.encode(elevation, design, hand_features)
        z1            = ElevationEncoder.reparameterize(mu, logvar)
        z_fused       = self.fuse(z1, c)
        x_recon       = self.decoder(z_fused, c)
        return x_recon, mu, logvar

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        design: torch.Tensor,
        hand_features: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate diverse elevation samples for a given design condition.

        Args:
            design       : (1, 1, H, W)  or (B, 1, H, W)
            hand_features: (1, HAND_FEATURE_DIM) or (B, HAND_FEATURE_DIM)
            num_samples  : K — number of samples to draw per design
            temperature  : scales the standard deviation of the z1 prior;
                           > 1.0 increases diversity, < 1.0 reduces it

        Returns:
            samples : (num_samples, 1, H, W)
        """
        self.eval()
        c = self.design_encoder(design, hand_features)          # (B_des, c_dim)
        # Expand c for all K samples (assumes single design, B_des=1)
        c_expanded = c.expand(num_samples, -1)                  # (K, c_dim)

        z1 = torch.randn(num_samples, self.z_dim, device=c.device) * temperature
        z_fused = self.fuse(z1, c_expanded)
        samples = self.decoder(z_fused, c_expanded)             # (K, 1, H, W)
        return samples

    @torch.no_grad()
    def reconstruct(
        self,
        elevation: torch.Tensor,
        design: torch.Tensor,
        hand_features: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic reconstruction (uses mu, not sampled z1)."""
        self.eval()
        mu, _, c = self.encode(elevation, design, hand_features)
        z_fused  = self.fuse(mu, c)
        return self.decoder(z_fused, c)
