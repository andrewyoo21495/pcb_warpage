#!/usr/bin/env python3
"""Conditional DDPM for PCB Warpage Distribution Generation.

Training flow:
  Design -> DDPMConditionEncoder -> (spatial_feats, global_cond)
  Elevation x0 -> add noise at random t -> x_t
  UNet(x_t, t, global_cond, spatial_feats) -> predicted noise
  Loss = MSE(predicted_noise, actual_noise)

Inference flow (DDIM):
  Design -> DDPMConditionEncoder -> (spatial_feats, global_cond)  [computed once]
  x_T ~ N(0, I)
  for t in reversed DDIM schedule:
      eps = UNet(x_t, t, global_cond, spatial_feats)
      x_{t-1} = DDIM_update(x_t, eps, t)
  return x_0

Config keys read:
    ddpm_T          int    (default 1000)  diffusion timesteps
    ddpm_ddim_steps int    (default 50)    DDIM inference steps
    ddpm_eta        float  (default 0.7)   DDIM stochasticity (0=deterministic, 1=full DDPM)
    ddpm_base_ch    int    (default 64)    U-Net base channels
    ddpm_cond_dim   int    (default 256)   global condition dimension
    ddpm_dropout    float  (default 0.15)  dropout rate
    image_size      int    (default 256)   spatial resolution
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ddpm_condition_encoder import DDPMConditionEncoder
from models.unet import UNet


def _cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule (Nichol & Dhariwal 2021).

    Returns:
        beta: (T,) float64
    """
    t = torch.arange(T + 1, dtype=torch.float64)
    f_t = torch.cos((t / T + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alpha_bar = f_t / f_t[0]
    beta = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return beta.clamp(max=0.999)


class ConditionalDDPM(nn.Module):
    """Conditional Denoising Diffusion Probabilistic Model.

    Provides the same sampling interface as CVAE:
        model.sample(design, hand_features, num_samples, temperature) -> (K, 1, H, W)

    Args:
        config: dict from load_config().
    """

    def __init__(self, config: dict):
        super().__init__()
        self.T = int(config.get('ddpm_t', 1000))
        self.ddim_steps = int(config.get('ddpm_ddim_steps', 50))
        self.eta = float(config.get('ddpm_eta', 0.7))
        self.image_size = int(config.get('image_size', 256))

        # Sub-networks
        self.cond_encoder = DDPMConditionEncoder(config)
        self.unet = UNet(config)

        # Precompute diffusion constants (registered as buffers for device transfer)
        beta = _cosine_beta_schedule(self.T)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer('beta', beta.float())
        self.register_buffer('alpha', alpha.float())
        self.register_buffer('alpha_bar', alpha_bar.float())
        self.register_buffer('sqrt_alpha_bar', alpha_bar.sqrt().float())
        self.register_buffer('sqrt_one_minus_alpha_bar', (1.0 - alpha_bar).sqrt().float())

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        elevation: torch.Tensor,
        design: torch.Tensor,
        hand_features: torch.Tensor,
    ) -> torch.Tensor:
        """Training forward: compute noise prediction loss.

        Internally normalizes elevation from [0,1] to [-1,1].

        Args:
            elevation     : (B, 1, H, W) float32 in [0, 1]
            design        : (B, 1, H, W) float32 in [0, 1]
            hand_features : (B, HAND_FEATURE_DIM)

        Returns:
            loss : scalar, MSE between predicted and actual noise
        """
        B = elevation.shape[0]
        device = elevation.device

        # Normalize to [-1, 1]
        x0 = elevation * 2.0 - 1.0

        # Random timesteps
        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)

        # Sample noise and create noisy image
        noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        x_t = sqrt_ab * x0 + sqrt_1mab * noise

        # Condition encoding
        spatial_feats, global_cond = self.cond_encoder(design, hand_features)

        # Predict noise
        noise_pred = self.unet(x_t, t, global_cond, spatial_feats)

        return F.mse_loss(noise_pred, noise)

    # ------------------------------------------------------------------
    # Inference (DDIM sampling)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        design: torch.Tensor,
        hand_features: torch.Tensor,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate elevation samples via DDIM reverse diffusion.

        Args:
            design        : (1, 1, H, W) or (B, 1, H, W)
            hand_features : (1, HAND_FEATURE_DIM) or (B, HAND_FEATURE_DIM)
            num_samples   : K samples to generate
            temperature   : scales DDIM eta; >1 = more diverse, <1 = less

        Returns:
            samples : (num_samples, 1, H, W) float32 in [0, 1]
        """
        self.eval()
        device = next(self.parameters()).device

        # Expand design for K samples
        design_exp = design[:1].expand(num_samples, -1, -1, -1)
        hand_exp = hand_features[:1].expand(num_samples, -1)

        # Compute condition once
        spatial_feats, global_cond = self.cond_encoder(design_exp, hand_exp)

        # Effective eta
        eta = min(self.eta * temperature, 1.0)

        # Build DDIM timestep schedule (evenly spaced, reversed)
        step_size = self.T // self.ddim_steps
        timesteps = list(range(0, self.T, step_size))  # ascending
        timesteps = list(reversed(timesteps))           # descending

        # Start from pure noise
        x = torch.randn(num_samples, 1, self.image_size, self.image_size, device=device)

        for i, t_val in enumerate(timesteps):
            t_batch = torch.full((num_samples,), t_val, device=device, dtype=torch.long)

            # Predict noise
            eps_pred = self.unet(x, t_batch, global_cond, spatial_feats)

            # Current and previous alpha_bar
            ab_t = self.alpha_bar[t_val]
            if i + 1 < len(timesteps):
                ab_t_prev = self.alpha_bar[timesteps[i + 1]]
            else:
                ab_t_prev = torch.tensor(1.0, device=device)

            # Predict x0
            x0_pred = (x - (1.0 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # DDIM sigma
            sigma = eta * (
                ((1.0 - ab_t_prev) / (1.0 - ab_t)).sqrt()
                * (1.0 - ab_t / ab_t_prev).sqrt()
            )

            # Direction pointing to x_t
            dir_xt = ((1.0 - ab_t_prev - sigma ** 2).clamp(min=0.0)).sqrt() * eps_pred

            # DDIM update
            if t_val > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = ab_t_prev.sqrt() * x0_pred + dir_xt + sigma * noise

        # Convert [-1, 1] -> [0, 1]
        return (x * 0.5 + 0.5).clamp(0.0, 1.0)
