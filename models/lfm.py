#!/usr/bin/env python3
"""Latent Flow Matching (LFM) for PCB Warpage Distribution Generation.

Two-stage approach (same Stage 1 as LDM):
  Stage 1 — Pretrained CVAE provides frozen encoder/decoder
  Stage 2 — Flow Matching in 64-dim latent space

Flow Matching learns an ODE velocity field v_θ(z_t, t, c) that transports
samples from a Gaussian prior z_0 ~ N(0,I) to the data distribution z_1 = mu.

Training:
    z_0 ~ N(0, I)                       (source noise)
    z_1 = ElevationEncoder(x).mu        (target latent, frozen)
    t ~ Uniform(0, 1)
    z_t = (1 - t) · z_0 + t · z_1       (linear interpolation / OT path)
    target_v = z_1 - z_0                 (constant velocity along straight line)
    loss = MSE(v_θ(z_t, t, c), target_v)

Inference:
    z_0 ~ N(0, I) · temperature
    Euler ODE:  z_{t+dt} = z_t + v_θ(z_t, t, c) · dt,  t: 0 → 1
    elevation = Decoder(Fuse(z_1, c), c)

Advantages over LDM:
  - No noise schedule (alpha_bar, beta, etc.) needed
  - Simpler loss landscape → potentially faster convergence
  - Fewer inference steps needed (straight ODE paths)

Config keys read:
    cvae_checkpoint        str   path to pretrained CVAE checkpoint
    lfm_ode_steps          int   (default 30)    ODE inference steps
    lfm_ode_solver         str   (default 'midpoint')  ODE solver: euler, midpoint, rk4
    lfm_hidden_dim         int   (default 512)   velocity net MLP width
    lfm_n_blocks           int   (default 8)     velocity net residual blocks
    lfm_dropout            float (default 0.1)   velocity net dropout
    lfm_sigma_min          float (default 0.001) numerical stability for t near 0/1
    lfm_finetune_encoder   bool  (default False) fine-tune design encoder
    z_dim                  int   (default 64)    latent dimension (from CVAE)
    c_dim                  int   (default 64)    condition dimension (from CVAE)
    fusion_method          str   (default 'film') fusion method (from CVAE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.design_encoder   import DesignEncoder
from models.elevation_encoder import ElevationEncoder
from models.decoder           import Decoder
from models.latent_denoiser   import LatentDenoiser


class LatentFlowMatching(nn.Module):
    """Latent Flow Matching — ODE-based generation in CVAE latent space.

    Provides the same sampling interface as CVAE and LDM:
        model.sample(design, hand_features, num_samples, temperature) -> (K, 1, H, W)

    Args:
        config: dict from load_config().
    """

    def __init__(self, config: dict):
        super().__init__()
        self.z_dim = int(config.get('z_dim', 64))
        self.c_dim = int(config.get('c_dim', 64))
        self.fusion_method = str(config.get('fusion_method', 'film'))
        self.finetune_encoder = bool(config.get('lfm_finetune_encoder', False))

        # Flow matching parameters
        self.ode_steps = int(config.get('lfm_ode_steps', 30))
        self.ode_solver = str(config.get('lfm_ode_solver', 'midpoint')).lower()
        self.sigma_min = float(config.get('lfm_sigma_min', 0.001))

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

        # ----- Velocity Network (the only trainable part) -----
        hidden_dim = int(config.get('lfm_hidden_dim', 512))
        n_blocks = int(config.get('lfm_n_blocks', 8))
        dropout = float(config.get('lfm_dropout', 0.1))

        self.velocity_net = LatentDenoiser(
            z_dim=self.z_dim,
            c_dim=self.c_dim,
            t_dim=128,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            dropout=dropout,
        )

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

        Args:
            cvae_checkpoint_path: path to the pretrained CVAE .pth checkpoint
        """
        print(f"Loading pretrained CVAE from: {cvae_checkpoint_path}")
        ckpt = torch.load(cvae_checkpoint_path, map_location='cpu',
                          weights_only=False)
        cvae_state = ckpt['model_state']

        own_state = self.state_dict()
        loaded_keys = []

        for key, value in cvae_state.items():
            if key in own_state and own_state[key].shape == value.shape:
                own_state[key] = value
                loaded_keys.append(key)

        self.load_state_dict(own_state, strict=False)
        print(f"  Loaded {len(loaded_keys)} weight tensors from CVAE checkpoint")

        self._freeze_cvae_components()

    def _freeze_cvae_components(self):
        """Freeze all CVAE-origin parameters; only velocity_net remains trainable."""
        for param in self.elevation_encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

        for name, param in self.named_parameters():
            if name.startswith(('film_gamma', 'film_beta',
                                'q_proj', 'kv_proj', 'cross_attn')):
                param.requires_grad = False

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
        """Training forward: compute flow matching velocity loss.

        Args:
            elevation     : (B, 1, H, W) float32 in [0, 1]
            design        : (B, 1, H, W) float32 in [0, 1]
            hand_features : (B, HAND_FEATURE_DIM)

        Returns:
            loss : scalar, MSE between predicted and target velocity
        """
        B = elevation.shape[0]
        device = elevation.device

        # Encode to latent (frozen)
        with torch.no_grad():
            mu, _ = self.elevation_encoder(elevation)
            z_1 = mu  # target (data point in latent space)

        # Condition vector
        if self.finetune_encoder:
            c = self.design_encoder(design, hand_features)
        else:
            with torch.no_grad():
                c = self.design_encoder(design, hand_features)

        # Source noise
        z_0 = torch.randn_like(z_1)

        # Random time t ~ Uniform(sigma_min, 1 - sigma_min)
        t = torch.rand(B, device=device) * (1.0 - 2 * self.sigma_min) + self.sigma_min

        # Linear interpolation (optimal transport path)
        t_expand = t.unsqueeze(1)  # (B, 1)
        z_t = (1.0 - t_expand) * z_0 + t_expand * z_1

        # Target velocity (constant along straight line)
        target_v = z_1 - z_0

        # Predict velocity
        v_pred = self.velocity_net(z_t, t, c)

        return F.mse_loss(v_pred, target_v)

    # ------------------------------------------------------------------
    # Inference (ODE integration in latent space)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        design: torch.Tensor,
        hand_features: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate elevation samples via ODE integration in latent space.

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
        c = self.design_encoder(design[:1], hand_features[:1])
        c_exp = c.expand(num_samples, -1)

        # Start from Gaussian noise
        z = torch.randn(num_samples, self.z_dim, device=device) * temperature

        # ODE integration: t = 0 → 1
        dt = 1.0 / self.ode_steps
        solver = self.ode_solver

        for i in range(self.ode_steps):
            t_i = i * dt

            if solver == 'midpoint':
                # Midpoint method (2nd-order): evaluate at half-step
                t_cur = torch.full((num_samples,), t_i, device=device)
                v1 = self.velocity_net(z, t_cur, c_exp)
                z_mid = z + v1 * (dt / 2)
                t_mid = torch.full((num_samples,), t_i + dt / 2, device=device)
                v_mid = self.velocity_net(z_mid, t_mid, c_exp)
                z = z + v_mid * dt

            elif solver == 'rk4':
                # Runge-Kutta 4th order
                t_cur = torch.full((num_samples,), t_i, device=device)
                t_half = torch.full((num_samples,), t_i + dt / 2, device=device)
                t_end = torch.full((num_samples,), t_i + dt, device=device)

                k1 = self.velocity_net(z, t_cur, c_exp)
                k2 = self.velocity_net(z + k1 * (dt / 2), t_half, c_exp)
                k3 = self.velocity_net(z + k2 * (dt / 2), t_half, c_exp)
                k4 = self.velocity_net(z + k3 * dt, t_end, c_exp)
                z = z + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)

            else:  # 'euler'
                t_cur = torch.full((num_samples,), t_i, device=device)
                v = self.velocity_net(z, t_cur, c_exp)
                z = z + v * dt

        # Decode: latent → elevation image
        z_fused = self.fuse(z, c_exp)
        samples = self.decoder(z_fused, c_exp)
        return samples.clamp(0.0, 1.0)
