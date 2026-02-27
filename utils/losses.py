#!/usr/bin/env python3
"""Loss functions for the CVAE PCB Warpage model.

Loss = Reconstruction Loss + beta * KL Divergence
  Reconstruction : MSE(x_reconstructed, x)
                   [+ spectral_weight * FFT-magnitude MSE if spectral_weight > 0]
  KL             : -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                 = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
                   [with per-dim free-bits clamping if free_bits > 0]
"""

import torch
import torch.nn.functional as F


def reconstruction_loss(x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean squared error between reconstruction and target."""
    return F.mse_loss(x_recon, x, reduction='mean')


def spectral_reconstruction_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """FFT magnitude loss to penalise frequency-domain blurring.

    Computes MSE between the 2-D DFT magnitude spectra of the reconstructed
    and real images.  Encourages the model to reproduce high-frequency spatial
    structure that pixel-wise MSE tends to smooth away.

    Args:
        x_recon : (B, 1, H, W)
        x       : (B, 1, H, W)

    Returns:
        scalar loss
    """
    fft_recon = torch.fft.rfft2(x_recon, norm='ortho')
    fft_real  = torch.fft.rfft2(x,       norm='ortho')
    return F.mse_loss(fft_recon.abs(), fft_real.abs())


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence between learned posterior N(mu, sigma^2) and prior N(0, I).

    Returns the mean over the batch and latent dimensions.
    Always used for logging; replaced by kl_divergence_free_bits in the
    actual loss when free_bits > 0.
    """
    kl = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )
    return kl


def kl_divergence_free_bits(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.5,
) -> torch.Tensor:
    """KL divergence with per-dimension free bits (Kingma et al. 2016).

    Computes per-dimension KL, averages over the batch, then clamps each
    dimension to at least `free_bits` nats before summing.  This prevents
    posterior collapse by guaranteeing the encoder keeps every latent
    dimension active — the decoder cannot ignore z1 entirely.

    A dimension whose posterior has drifted to the prior (mu≈0, sigma≈1)
    contributes ~0 nats of raw KL.  Clamping at free_bits forces the
    optimiser to re-engage those dimensions rather than zeroing them out.

    Args:
        mu        : (B, z_dim) posterior mean
        logvar    : (B, z_dim) posterior log-variance
        free_bits : minimum KL per latent dimension (nats).
                    Typical range: 0.1 – 2.0.  0 = standard KL (no clamping).

    Returns:
        scalar — on the same scale as kl_divergence().
    """
    # Per-sample, per-dim KL: (B, z_dim)
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    # Mean over batch first, then clamp per dim, then sum over dims
    kl_mean = kl_per_dim.mean(dim=0)        # (z_dim,)
    kl_free = kl_mean.clamp(min=free_bits)  # (z_dim,)
    return kl_free.sum()


def cvae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    free_bits: float = 0.0,
    spectral_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined CVAE loss with optional free-bits KL and spectral reconstruction.

    Args:
        x_recon        : Reconstructed elevation image  (B, 1, H, W)
        x              : Target elevation image          (B, 1, H, W)
        mu             : Posterior mean                  (B, z_dim)
        logvar         : Posterior log-variance          (B, z_dim)
        beta           : KL weight (annealed during training)
        free_bits      : Minimum KL per latent dim (nats). 0 = disabled.
        spectral_weight: Weight for FFT magnitude loss.  0 = disabled.

    Returns:
        (total_loss, recon_loss, kl_raw)
        kl_raw is always the standard (no free-bits) KL for logging purposes.
    """
    recon = reconstruction_loss(x_recon, x)
    if spectral_weight > 0.0:
        recon = recon + spectral_weight * spectral_reconstruction_loss(x_recon, x)

    # Raw KL for logging (always computed)
    kl_raw = kl_divergence(mu, logvar)

    # KL used in the actual loss (with or without free bits)
    kl_penalised = (kl_divergence_free_bits(mu, logvar, free_bits)
                    if free_bits > 0.0 else kl_raw)

    total = recon + beta * kl_penalised
    return total, recon, kl_raw


def get_cyclical_beta(
    epoch: int,
    total_epochs: int,
    beta_max: float,
    n_cycles: int,
) -> float:
    """Cyclical KL annealing schedule.

    Resets beta to 0 at the start of each cycle and linearly ramps to
    beta_max by the end of the cycle (preventing posterior collapse).

    Args:
        epoch        : Current epoch (0-indexed)
        total_epochs : Total training epochs
        beta_max     : Maximum beta value
        n_cycles     : Number of complete annealing cycles

    Returns:
        beta value for the current epoch
    """
    cycle_len = total_epochs / n_cycles
    pos_in_cycle = epoch % cycle_len
    beta = beta_max * (pos_in_cycle / cycle_len)
    return float(beta)
