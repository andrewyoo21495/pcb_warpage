#!/usr/bin/env python3
"""Latent Diffusion Model (LDM) for PCB Warpage Distribution Generation.

Two-stage approach:
  Stage 1 — Pretrained CVAE provides frozen encoder/decoder:
    ElevationEncoder : elevation image → (mu, logvar)  → z_0 = mu
    DesignEncoder    : design image → c
    Decoder          : (z_fused, c) → reconstructed elevation

  Stage 2 — Lightweight MLP diffusion in 64-dim latent space:
    Forward:  z_t = √ᾱ_t · z_0 + √(1−ᾱ_t) · ε
    Reverse:  ε_θ(z_t, t, c) predicts noise via LatentDenoiser
    Sampling: DDIM in latent space → decoder → elevation image

Only the LatentDenoiser is trained; all CVAE components are frozen.

Config keys read:
    cvae_checkpoint       str   path to pretrained CVAE checkpoint
    ldm_T                 int   (default 500)   diffusion timesteps in latent space
    ldm_ddim_steps        int   (default 50)    DDIM inference steps
    ldm_hidden_dim        int   (default 512)   denoiser MLP width
    ldm_n_blocks          int   (default 8)     denoiser residual blocks
    ldm_dropout           float (default 0.1)   denoiser dropout
    ldm_finetune_encoder  bool  (default False)  fine-tune design encoder
    z_dim                 int   (default 64)    latent dimension (from CVAE)
    c_dim                 int   (default 64)    condition dimension (from CVAE)
    fusion_method         str   (default 'film') fusion method (from CVAE)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.design_encoder   import DesignEncoder
from models.elevation_encoder import ElevationEncoder
from models.decoder           import Decoder
from models.latent_denoiser   import LatentDenoiser


def _cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule (Nichol & Dhariwal 2021)."""
    t = torch.arange(T + 1, dtype=torch.float64)
    f_t = torch.cos((t / T + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alpha_bar = f_t / f_t[0]
    beta = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return beta.clamp(max=0.999)


class LatentDiffusionModel(nn.Module):
    """Latent Diffusion Model — DDIM diffusion in CVAE latent space.

    Provides the same sampling interface as CVAE and DDPM:
        model.sample(design, hand_features, num_samples, temperature) -> (K, 1, H, W)

    Args:
        config: dict from load_config().
    """

    def __init__(self, config: dict):
        super().__init__()
        self.z_dim = int(config.get('z_dim', 64))
        self.c_dim = int(config.get('c_dim', 64))
        self.fusion_method = str(config.get('fusion_method', 'film'))
        self.finetune_encoder = bool(config.get('ldm_finetune_encoder', False))

        # Diffusion parameters
        self.T = int(config.get('ldm_t', 500))
        self.ddim_steps = int(config.get('ldm_ddim_steps', 50))

        # ----- CVAE components (loaded later via load_pretrained_cvae) -----
        self.design_encoder = DesignEncoder(config)
        self.elevation_encoder = ElevationEncoder(config)

        # Fused latent dimension (must match CVAE)
        if self.fusion_method == 'concat':
            z_fused_dim = self.z_dim + self.c_dim
        else:
            z_fused_dim = self.z_dim

        # Fusion layers (must match CVAE)
        if self.fusion_method == 'film':
            self.film_gamma = nn.Linear(self.c_dim, self.z_dim)
            self.film_beta = nn.Linear(self.c_dim, self.z_dim)
        elif self.fusion_method == 'cross_attention':
            embed_dim = self.z_dim
            self.n_tokens = 4
            self.q_proj = nn.Linear(self.c_dim, embed_dim)
            self.kv_proj = nn.Linear(self.z_dim // self.n_tokens, embed_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=4, batch_first=True)

        self.decoder = Decoder(config, z_fused_dim)

        # ----- Latent Denoiser (the only trainable part) -----
        hidden_dim = int(config.get('ldm_hidden_dim', 512))
        n_blocks = int(config.get('ldm_n_blocks', 8))
        dropout = float(config.get('ldm_dropout', 0.1))

        self.denoiser = LatentDenoiser(
            z_dim=self.z_dim,
            c_dim=self.c_dim,
            t_dim=128,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            dropout=dropout,
        )

        # ----- Precompute diffusion schedule -----
        beta = _cosine_beta_schedule(self.T)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer('alpha_bar', alpha_bar.float())
        self.register_buffer('sqrt_alpha_bar', alpha_bar.sqrt().float())
        self.register_buffer('sqrt_one_minus_alpha_bar',
                             (1.0 - alpha_bar).sqrt().float())

    # ------------------------------------------------------------------
    # Fusion (same logic as CVAE)
    # ------------------------------------------------------------------

    def fuse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Fuse latent z with condition c (same method as pretrained CVAE)."""
        if self.fusion_method == 'concat':
            return torch.cat([z, c], dim=1)
        elif self.fusion_method == 'film':
            gamma = self.film_gamma(c)
            beta = self.film_beta(c)
            return gamma * z + beta
        elif self.fusion_method == 'cross_attention':
            B = z.size(0)
            Q = self.q_proj(c).unsqueeze(1)
            z_tok = z.view(B, self.n_tokens, -1)
            K = self.kv_proj(z_tok)
            V = K
            out, _ = self.cross_attn(Q, K, V)
            return out.squeeze(1)
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method!r}")

    # ------------------------------------------------------------------
    # Pretrained CVAE loading
    # ------------------------------------------------------------------

    def load_pretrained_cvae(self, cvae_checkpoint_path: str):
        """Load pretrained CVAE weights and freeze encoder/decoder.

        The CVAE checkpoint must contain 'model_state' with the CVAE state dict.
        Weights are mapped to the corresponding sub-modules in this model.

        Args:
            cvae_checkpoint_path: path to the pretrained CVAE .pth checkpoint
        """
        print(f"Loading pretrained CVAE from: {cvae_checkpoint_path}")
        ckpt = torch.load(cvae_checkpoint_path, map_location='cpu',
                          weights_only=False)
        cvae_state = ckpt['model_state']

        # Map CVAE state dict keys to our module names
        # CVAE keys: design_encoder.*, elevation_encoder.*, decoder.*,
        #            film_gamma.*, film_beta.* (or cross_attention.*)
        own_state = self.state_dict()
        loaded_keys = []

        for key, value in cvae_state.items():
            if key in own_state and own_state[key].shape == value.shape:
                own_state[key] = value
                loaded_keys.append(key)

        self.load_state_dict(own_state, strict=False)
        print(f"  Loaded {len(loaded_keys)} weight tensors from CVAE checkpoint")

        # Freeze CVAE components — only denoiser should be trainable
        self._freeze_cvae_components()

    def _freeze_cvae_components(self):
        """Freeze all CVAE-origin parameters; only denoiser remains trainable."""
        # Always freeze elevation encoder and decoder
        for param in self.elevation_encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

        # Freeze fusion layers
        for name, param in self.named_parameters():
            if name.startswith(('film_gamma', 'film_beta',
                                'q_proj', 'kv_proj', 'cross_attn')):
                param.requires_grad = False

        # Design encoder: optionally fine-tune
        if not self.finetune_encoder:
            for param in self.design_encoder.parameters():
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} parameters "
              f"(design_encoder finetune={self.finetune_encoder})")

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        elevation: torch.Tensor,
        design: torch.Tensor,
        hand_features: torch.Tensor,
    ) -> torch.Tensor:
        """Training forward: compute noise prediction loss in latent space.

        Args:
            elevation     : (B, 1, H, W) float32 in [0, 1]
            design        : (B, 1, H, W) float32 in [0, 1]
            hand_features : (B, HAND_FEATURE_DIM)

        Returns:
            loss : scalar, MSE between predicted and actual noise in latent space
        """
        B = elevation.shape[0]
        device = elevation.device

        # Encode to latent (frozen)
        with torch.no_grad():
            mu, _ = self.elevation_encoder(elevation)
            z_0 = mu  # deterministic — diffusion provides stochasticity

        # Condition vector
        if self.finetune_encoder:
            c = self.design_encoder(design, hand_features)
        else:
            with torch.no_grad():
                c = self.design_encoder(design, hand_features)

        # Random timesteps
        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)

        # Forward diffusion in latent space
        noise = torch.randn_like(z_0)
        sqrt_ab = self.sqrt_alpha_bar[t]      # (B,)
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t]  # (B,)

        z_t = sqrt_ab.unsqueeze(1) * z_0 + sqrt_1mab.unsqueeze(1) * noise

        # Predict noise
        noise_pred = self.denoiser(z_t, t, c)

        return F.mse_loss(noise_pred, noise)

    # ------------------------------------------------------------------
    # Inference (DDIM sampling in latent space)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        design: torch.Tensor,
        hand_features: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate elevation samples via DDIM in latent space + CVAE decoder.

        Args:
            design        : (1, 1, H, W) or (B, 1, H, W)
            hand_features : (1, HAND_FEATURE_DIM) or (B, HAND_FEATURE_DIM)
            num_samples   : K samples to generate
            temperature   : scales initial noise std; >1 = more diverse

        Returns:
            samples : (num_samples, 1, H, W) float32 in [0, 1]
        """
        self.eval()
        device = next(self.parameters()).device

        # Condition
        c = self.design_encoder(design[:1], hand_features[:1])  # (1, c_dim)
        c_exp = c.expand(num_samples, -1)  # (K, c_dim)

        # DDIM schedule (evenly spaced, descending)
        step_size = max(self.T // self.ddim_steps, 1)
        timesteps = list(range(0, self.T, step_size))
        timesteps = list(reversed(timesteps))

        # Start from Gaussian noise in latent space
        z = torch.randn(num_samples, self.z_dim, device=device) * temperature

        # DDIM reverse loop
        eta = 0.0  # deterministic DDIM for latent space (cleaner samples)

        for i, t_val in enumerate(timesteps):
            t_batch = torch.full((num_samples,), t_val, device=device,
                                 dtype=torch.long)

            # Predict noise
            eps_pred = self.denoiser(z, t_batch, c_exp)

            # Current and previous alpha_bar
            ab_t = self.alpha_bar[t_val]
            if i + 1 < len(timesteps):
                ab_t_prev = self.alpha_bar[timesteps[i + 1]]
            else:
                ab_t_prev = torch.tensor(1.0, device=device)

            # Predict z_0
            z0_pred = (z - (1.0 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()

            # DDIM sigma
            sigma = eta * (
                ((1.0 - ab_t_prev) / (1.0 - ab_t)).sqrt()
                * (1.0 - ab_t / ab_t_prev).sqrt()
            )

            # Direction
            dir_zt = ((1.0 - ab_t_prev - sigma ** 2).clamp(min=0.0)).sqrt() * eps_pred

            # Update
            if t_val > 0:
                noise = torch.randn_like(z)
            else:
                noise = torch.zeros_like(z)

            z = ab_t_prev.sqrt() * z0_pred + dir_zt + sigma * noise

        # Decode: latent → elevation image
        z_fused = self.fuse(z, c_exp)
        samples = self.decoder(z_fused, c_exp)  # (K, 1, H, W) in [0,1]
        return samples.clamp(0.0, 1.0)
