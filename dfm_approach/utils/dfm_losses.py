#!/usr/bin/env python3
"""Loss functions for DF²M training.

Module A losses:
    - MSE on mean warpage
    - Laplacian smoothness (physics-informed)
    - Spectral (FFT magnitude) loss

Module B losses:
    - Residual reconstruction MSE
    - KL divergence with soft free bits
    - Spectral reconstruction loss
"""

import torch
import torch.nn.functional as F


# ==================================================================
# Module A: FNO losses
# ==================================================================

def laplacian_smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    """Penalise non-smooth regions via discrete Laplacian.

    L_smooth = mean(|∇²x|²)

    Encourages physically plausible smooth deformation fields.

    Args:
        x: (B, 1, H, W) predicted mean warpage
    """
    # 2D Laplacian kernel
    kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]], dtype=x.dtype, device=x.device
    ).view(1, 1, 3, 3)

    lap = F.conv2d(x.float(), kernel, padding=1)
    return (lap ** 2).mean()


def spectral_loss(x_pred: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
    """FFT magnitude loss for frequency-domain fidelity.

    Args:
        x_pred:   (B, 1, H, W) predicted
        x_target: (B, 1, H, W) target
    """
    # Disable autocast: FFT produces complex tensors incompatible with AMP
    device_type = 'cuda' if x_pred.device.type == 'cuda' else 'cpu'
    with torch.amp.autocast(device_type, enabled=False):
        fft_pred = torch.fft.rfft2(x_pred.float(), norm='ortho')
        fft_target = torch.fft.rfft2(x_target.float(), norm='ortho')
        return F.mse_loss(fft_pred.abs(), fft_target.abs())


def fno_loss(
    mean_pred: torch.Tensor,
    mean_target: torch.Tensor,
    smooth_weight: float = 0.01,
    spectral_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined FNO loss.

    L = MSE + λ_smooth · Laplacian + λ_spectral · SpectralLoss

    Returns:
        (total_loss, metrics_dict)
    """
    mse = F.mse_loss(mean_pred, mean_target)
    l_smooth = laplacian_smoothness_loss(mean_pred)
    l_spectral = spectral_loss(mean_pred, mean_target)

    total = mse + smooth_weight * l_smooth + spectral_weight * l_spectral

    metrics = {
        'mse': mse.item(),
        'smooth': l_smooth.item(),
        'spectral': l_spectral.item(),
        'total': total.item(),
    }
    return total, metrics


# ==================================================================
# Module B: CAE losses
# ==================================================================

def kl_divergence_soft_free_bits(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KL divergence with differentiable soft free bits.

    Instead of hard clamp(KL_dim, min=free_bits), uses a soft penalty:
        KL_eff_dim = KL_dim + free_bits · softplus(free_bits - KL_dim)

    This is differentiable everywhere and gently pushes KL above the
    free_bits threshold without hard discontinuities.

    Args:
        mu:        (B, z_dim)
        logvar:    (B, z_dim)
        free_bits: minimum KL per dim (nats)

    Returns:
        kl_loss:   scalar (used in optimisation)
        kl_raw:    scalar (standard KL for logging)
    """
    # Per-sample, per-dim KL
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # (B, z_dim)

    # Mean over batch
    kl_mean = kl_per_dim.mean(dim=0)  # (z_dim,)

    # Standard KL for logging
    kl_raw = kl_mean.sum()

    if free_bits > 0:
        # Soft free bits: penalty when KL drops below threshold
        deficit = free_bits - kl_mean
        soft_penalty = free_bits * F.softplus(deficit)
        kl_loss = (kl_mean + soft_penalty).sum()
    else:
        kl_loss = kl_raw

    return kl_loss, kl_raw


def cae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    free_bits: float = 0.5,
    spectral_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined Residual CAE loss.

    L = MSE(recon, target) + β · KL + λ_spectral · SpectralLoss

    Returns:
        (total_loss, metrics_dict)
    """
    recon_mse = F.mse_loss(recon, target)
    kl_loss, kl_raw = kl_divergence_soft_free_bits(mu, logvar, free_bits)
    l_spectral = spectral_loss(recon, target)

    total = recon_mse + beta * kl_loss + spectral_weight * l_spectral

    # Active KL dimensions (for monitoring)
    with torch.no_grad():
        kl_per_dim = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)
        active_dims = (kl_per_dim > 0.1).sum().item()

    metrics = {
        'recon_mse': recon_mse.item(),
        'kl_raw': kl_raw.item(),
        'kl_loss': kl_loss.item(),
        'spectral': l_spectral.item(),
        'total': total.item(),
        'active_kl_dims': active_dims,
        'beta': beta,
    }
    return total, metrics


def get_cyclical_beta(
    epoch: int,
    total_epochs: int,
    beta_max: float,
    n_cycles: int,
    warmup_epochs: int = 10,
) -> float:
    """Cyclical KL annealing with warmup.

    First `warmup_epochs` epochs use β=0 (reconstruction-only warmup).
    Then cyclical annealing begins.

    Args:
        epoch:          current epoch (0-indexed)
        total_epochs:   total training epochs
        beta_max:       maximum β value
        n_cycles:       number of annealing cycles
        warmup_epochs:  pure reconstruction warmup (β=0)
    """
    if epoch < warmup_epochs:
        return 0.0

    adjusted_epoch = epoch - warmup_epochs
    adjusted_total = total_epochs - warmup_epochs
    if adjusted_total <= 0:
        return beta_max

    cycle_len = adjusted_total / n_cycles
    pos_in_cycle = adjusted_epoch % cycle_len
    return float(beta_max * (pos_in_cycle / cycle_len))
