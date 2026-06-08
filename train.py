#!/usr/bin/env python3
"""Training script for PCB Warpage models (CVAE, LDM, LFM).

Usage:
  python train.py                      # uses config.txt in current dir
  python train.py --config config.txt  # explicit path
  python train.py --val_fold 1         # override leave-one-out fold

Set model_type in config.txt:
  model_type  cvae   -> Conditional VAE with cyclical KL annealing
  model_type  ldm    -> Latent Diffusion Model (requires pretrained CVAE)
  model_type  lfm    -> Latent Flow Matching (requires pretrained CVAE)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

from utils.load_config import load_config, display_config
from utils.dataset     import create_dataloaders
from utils.losses      import cvae_loss, get_cyclical_beta
from models            import build_model


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Train PCB Warpage model')
    parser.add_argument('--config',   type=str, default='config.txt')
    parser.add_argument('--val_fold', type=int, default=None,
                        help='Override val_fold from config (0-indexed)')
    parser.add_argument('--gpu',      type=int, default=None,
                        help='Override gpu_ids from config (e.g. --gpu 3)')
    parser.add_argument('--tag',      type=str, default=None,
                        help='Tag appended to output paths for parallel runs '
                             '(e.g. --tag fold0 → outputs/cvae_pcb_fold0.pth)')
    parser.add_argument('--cvae-checkpoint', type=str, default=None,
                        help='Override cvae_checkpoint for LDM/LFM training '
                             '(e.g. --cvae-checkpoint outputs/cvae_pcb_fold0.pth)')
    return parser.parse_args()


# ------------------------------------------------------------------
# Device setup
# ------------------------------------------------------------------

def get_device(config: dict) -> torch.device:
    gpu_ids = config.get('gpu_ids', -1)
    if isinstance(gpu_ids, list):
        gpu_id = gpu_ids[0]
    else:
        gpu_id = int(gpu_ids)

    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


# ------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------

def setup_logger(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('PCB_Warpage')
    logger.setLevel(logging.INFO)
    # Clear any existing handlers (avoids duplicate logs on re-run)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ==================================================================
# CVAE training functions
# ==================================================================

def _count_active_kl_dims(
    kl_per_dim_acc: torch.Tensor,
    n_batches: int,
    threshold: float = 0.1,
) -> int:
    """Return number of latent dims whose mean KL exceeds threshold.

    A dim is 'active' (not collapsed) when the encoder has learned to
    encode real information there rather than defaulting to the prior.
    """
    mean_kl = kl_per_dim_acc / max(n_batches, 1)  # (z_dim,)
    return int((mean_kl > threshold).sum().item())


def train_one_epoch_cvae(model, loader, optimizer, scaler, device, use_amp, beta,
                         free_bits, spectral_weight, aux_weight):
    model.train()
    total_loss = recon_sum = kl_sum = 0.0
    kl_per_dim_acc = None
    n_batches = 0

    for design, elevation, hand_features in loader:
        design        = design.to(device, non_blocking=True)
        elevation     = elevation.to(device, non_blocking=True)
        hand_features = hand_features.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            x_recon, mu, logvar, x_recon_aux = model(elevation, design, hand_features)
            loss, recon, kl = cvae_loss(
                x_recon, elevation, mu, logvar, beta,
                free_bits=free_bits, spectral_weight=spectral_weight,
                x_recon_aux=x_recon_aux, aux_weight=aux_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        recon_sum  += recon.item()
        kl_sum     += kl.item()

        # Accumulate per-dim KL for active dim counting (detached, no grad)
        with torch.no_grad():
            kl_dims = -0.5 * (
                1 + logvar.detach() - mu.detach().pow(2) - logvar.detach().exp()
            ).mean(dim=0)
            kl_per_dim_acc = kl_dims if kl_per_dim_acc is None else kl_per_dim_acc + kl_dims

        n_batches += 1

    n = max(n_batches, 1)
    active_dims = (_count_active_kl_dims(kl_per_dim_acc, n_batches)
                   if kl_per_dim_acc is not None else 0)
    return total_loss / n, recon_sum / n, kl_sum / n, active_dims


@torch.no_grad()
def validate_cvae(model, loader, device, use_amp, beta, free_bits, spectral_weight,
                  aux_weight):
    model.eval()
    total_loss = recon_sum = kl_sum = 0.0
    kl_per_dim_acc = None
    n_batches = 0

    for design, elevation, hand_features in loader:
        design        = design.to(device, non_blocking=True)
        elevation     = elevation.to(device, non_blocking=True)
        hand_features = hand_features.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            x_recon, mu, logvar, x_recon_aux = model(elevation, design, hand_features)
            loss, recon, kl = cvae_loss(
                x_recon, elevation, mu, logvar, beta,
                free_bits=free_bits, spectral_weight=spectral_weight,
                x_recon_aux=x_recon_aux, aux_weight=aux_weight)

        total_loss += loss.item()
        recon_sum  += recon.item()
        kl_sum     += kl.item()

        kl_dims = -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        ).mean(dim=0)
        kl_per_dim_acc = kl_dims if kl_per_dim_acc is None else kl_per_dim_acc + kl_dims

        n_batches += 1

    n = max(n_batches, 1)
    active_dims = (_count_active_kl_dims(kl_per_dim_acc, n_batches)
                   if kl_per_dim_acc is not None else 0)
    return total_loss / n, recon_sum / n, kl_sum / n, active_dims


# ==================================================================
# LDM / LFM training functions (shared — both return scalar loss)
# ==================================================================

def train_one_epoch_latent(model, ema, loader, optimizer, scaler, device, use_amp):
    """One training epoch for LDM or LFM (identical interface: forward → scalar loss)."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for design, elevation, hand_features in loader:
        design        = design.to(device, non_blocking=True)
        elevation     = elevation.to(device, non_blocking=True)
        hand_features = hand_features.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss = model(elevation, design, hand_features)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_latent(model, loader, device, use_amp):
    """Validation for LDM or LFM."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for design, elevation, hand_features in loader:
        design        = design.to(device, non_blocking=True)
        elevation     = elevation.to(device, non_blocking=True)
        hand_features = hand_features.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss = model(elevation, design, hand_features)
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args   = parse_args()
    config = load_config(args.config)

    if args.val_fold is not None:
        config['val_fold'] = args.val_fold
    if args.gpu is not None:
        config['gpu_ids'] = args.gpu
    if args.cvae_checkpoint is not None:
        config['cvae_checkpoint'] = args.cvae_checkpoint

    # --tag: append tag to output paths so parallel runs don't collide
    if args.tag:
        tag = args.tag
        for key in ('modelpath', 'log_file_dir', 'vis_save_dir', 'sample_save_dir'):
            if key in config:
                p = Path(config[key])
                if p.suffix:  # file path  (e.g. model.pth, train.log)
                    config[key] = str(p.with_stem(f"{p.stem}_{tag}"))
                else:         # directory  (e.g. outputs/vis_cvae)
                    config[key] = str(p.parent / f"{p.name}_{tag}")

    display_config(config)

    model_type = str(config.get('model_type', 'cvae')).lower()

    # Logger
    log_path = config.get('log_file_dir', './outputs/train.log')
    logger   = setup_logger(log_path)

    # Device
    device = get_device(config)
    use_amp = device.type == 'cuda'

    # Data
    train_loader, val_loader = create_dataloaders(config)

    # Model
    model = build_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model type: {model_type.upper()}  |  Parameters: {total_params:,}")

    # Shared training parameters
    lr           = float(config.get('learning_rate', 1e-4))
    weight_decay = float(config.get('weight_decay',  1e-4))
    total_epochs = int(config.get('training_epochs', 200))

    # Early-stopping threshold
    early_stop_thresh = float(config.get('early_stop_threshold', 0.0))

    # Checkpoint
    model_path = config.get('modelpath', './outputs/cvae_pcb.pth')
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')

    stop_info = (f"  |  early_stop_threshold={early_stop_thresh:.4f}"
                 if early_stop_thresh > 0.0 else "")

    # ============================================================
    # CVAE training
    # ============================================================
    if model_type == 'cvae':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        beta_max        = float(config.get('beta_max',        0.2))
        beta_cycles     = int(config.get('beta_cycles',       4))
        free_bits       = float(config.get('free_bits',       0.5))
        spectral_weight = float(config.get('spectral_weight', 0.1))
        aux_weight      = float(config.get('aux_weight',      0.0))
        z_dim           = int(config.get('z_dim', 64))

        logger.info("=" * 60)
        logger.info(f"Training CVAE  |  fusion={config.get('fusion_method')}  "
                    f"|  val_fold={config.get('val_fold')}  "
                    f"|  epochs={total_epochs}  "
                    f"|  free_bits={free_bits}  spectral_weight={spectral_weight}  "
                    f"aux_weight={aux_weight}{stop_info}")
        logger.info("=" * 60)

        for epoch in range(total_epochs):
            beta = get_cyclical_beta(epoch, total_epochs, beta_max, beta_cycles)

            t0 = time.time()
            train_loss, train_recon, train_kl, train_active = train_one_epoch_cvae(
                model, train_loader, optimizer, scaler, device, use_amp, beta,
                free_bits, spectral_weight, aux_weight)
            val_loss, val_recon, val_kl, val_active = validate_cvae(
                model, val_loader, device, use_amp, beta, free_bits, spectral_weight,
                aux_weight)
            scheduler.step()
            elapsed = time.time() - t0

            logger.info(
                f"Epoch {epoch+1:4d}/{total_epochs}  "
                f"beta={beta:.3f}  "
                f"train[loss={train_loss:.4f} recon={train_recon:.4f} "
                f"kl={train_kl:.4f} active={train_active}/{z_dim}]  "
                f"val[loss={val_loss:.4f} recon={val_recon:.4f} "
                f"kl={val_kl:.4f} active={val_active}/{z_dim}]  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"({elapsed:.1f}s)"
            )

            if val_recon < best_val_loss:
                best_val_loss = val_recon
                torch.save({
                    'epoch':          epoch + 1,
                    'model_type':     'cvae',
                    'model_state':    model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_loss':       val_loss,
                    'val_recon':      val_recon,
                    'config':         config,
                }, model_path)
                logger.info(f"  -> Checkpoint saved (val_recon={val_recon:.4f})")

            if early_stop_thresh > 0.0 and val_recon < early_stop_thresh:
                logger.info(
                    f"Early stop at epoch {epoch+1}: "
                    f"val_recon={val_recon:.4f} < threshold={early_stop_thresh:.4f}"
                )
                break

        logger.info("Training complete.")
        logger.info(f"Best val recon loss: {best_val_loss:.4f}")

    # ============================================================
    # LDM / LFM training (shared logic — both use latent-space loss)
    # ============================================================
    elif model_type in ('ldm', 'lfm'):
        from utils.ema import EMA

        # Load pretrained CVAE weights
        cvae_path = str(config.get('cvae_checkpoint', './outputs/cvae_pcb.pth'))
        if not Path(cvae_path).exists():
            raise FileNotFoundError(
                f"Pretrained CVAE checkpoint not found: {cvae_path}\n"
                f"Train a CVAE first (model_type=cvae), then set "
                f"'cvae_checkpoint' in your config to point to the CVAE .pth file."
            )
        model.load_pretrained_cvae(cvae_path)

        # Only optimize trainable parameters (denoiser / velocity_net)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        ema_decay = float(config.get('ema_decay', 0.9999))
        ema = EMA(model, decay=ema_decay)

        model_label = model_type.upper()
        loss_label = 'noise-pred' if model_type == 'ldm' else 'velocity'

        logger.info("=" * 60)
        logger.info(f"Training {model_label}  |  cvae={cvae_path}  "
                    f"|  val_fold={config.get('val_fold')}  "
                    f"|  epochs={total_epochs}  "
                    f"|  ema_decay={ema_decay}{stop_info}")
        logger.info("=" * 60)

        for epoch in range(total_epochs):
            t0 = time.time()
            train_loss = train_one_epoch_latent(
                model, ema, train_loader, optimizer, scaler, device, use_amp)
            val_loss = validate_latent(model, val_loader, device, use_amp)
            scheduler.step()
            elapsed = time.time() - t0

            logger.info(
                f"Epoch {epoch+1:4d}/{total_epochs}  "
                f"train_{loss_label}={train_loss:.6f}  "
                f"val_{loss_label}={val_loss:.6f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"({elapsed:.1f}s)"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch':             epoch + 1,
                    'model_type':        model_type,
                    'model_state':       model.state_dict(),
                    'ema_state_dict':    ema.shadow,
                    'optimizer_state':   optimizer.state_dict(),
                    'val_loss':          val_loss,
                    'config':            config,
                    'cvae_checkpoint':   cvae_path,
                }, model_path)
                logger.info(f"  -> Checkpoint saved (val_{loss_label}={val_loss:.6f})")

            if early_stop_thresh > 0.0 and val_loss < early_stop_thresh:
                logger.info(
                    f"Early stop at epoch {epoch+1}: "
                    f"val_{loss_label}={val_loss:.6f} < threshold={early_stop_thresh:.4f}"
                )
                break

        logger.info("Training complete.")
        logger.info(f"Best val {loss_label} loss: {best_val_loss:.6f}")

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    logger.info(f"Model saved to: {model_path}")


if __name__ == '__main__':
    main()
