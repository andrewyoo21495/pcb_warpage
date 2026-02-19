#!/usr/bin/env python3
"""Loss functions for the CVAE PCB Warpage model.

Loss = Reconstruction Loss + beta * KL Divergence
  Reconstruction : MSE(x_reconstructed, x)
  KL             : -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                 = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
"""

import torch
import torch.nn.functional as F


def reconstruction_loss(x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean squared error between reconstruction and target."""
    return F.mse_loss(x_recon, x, reduction='mean')


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence between learned posterior N(mu, sigma^2) and prior N(0, I).

    Returns the mean over the batch (and latent dimensions).
    """
    # Sum over latent dim, mean over batch
    kl = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )
    return kl


def cvae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined CVAE loss.

    Args:
        x_recon : Reconstructed elevation image  (B, 1, H, W)
        x       : Target elevation image          (B, 1, H, W)
        mu      : Posterior mean                  (B, z_dim)
        logvar  : Posterior log-variance          (B, z_dim)
        beta    : KL weight (annealed during training)

    Returns:
        (total_loss, recon_loss, kl_loss)
    """
    recon = reconstruction_loss(x_recon, x)
    kl    = kl_divergence(mu, logvar)
    total = recon + beta * kl
    return total, recon, kl


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
