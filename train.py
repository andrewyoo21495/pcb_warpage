#!/usr/bin/env python3
"""Training script for PCB Warpage models (CVAE and DDPM).

Usage:
  python train.py                      # uses config.txt in current dir
  python train.py --config config.txt  # explicit path
  python train.py --val_fold 1         # override leave-one-out fold

Set model_type in config.txt:
  model_type  cvae   -> Conditional VAE with cyclical KL annealing
  model_type  ddpm   -> Conditional DDPM with EMA
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

def train_one_epoch_cvae(model, loader, optimizer, device, beta):
    model.train()
    total_loss = recon_sum = kl_sum = 0.0
    n_batches = 0

    for design, elevation, hand_features in loader:
        design        = design.to(device)
        elevation     = elevation.to(device)
        hand_features = hand_features.to(device)

        optimizer.zero_grad()
        x_recon, mu, logvar = model(elevation, design, hand_features)
        loss, recon, kl     = cvae_loss(x_recon, elevation, mu, logvar, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        recon_sum  += recon.item()
        kl_sum     += kl.item()
        n_batches  += 1

    n = max(n_batches, 1)
    return total_loss / n, recon_sum / n, kl_sum / n


@torch.no_grad()
def validate_cvae(model, loader, device, beta):
    model.eval()
    total_loss = recon_sum = kl_sum = 0.0
    n_batches = 0

    for design, elevation, hand_features in loader:
        design        = design.to(device)
        elevation     = elevation.to(device)
        hand_features = hand_features.to(device)

        x_recon, mu, logvar = model(elevation, design, hand_features)
        loss, recon, kl     = cvae_loss(x_recon, elevation, mu, logvar, beta)

        total_loss += loss.item()
        recon_sum  += recon.item()
        kl_sum     += kl.item()
        n_batches  += 1

    n = max(n_batches, 1)
    return total_loss / n, recon_sum / n, kl_sum / n


# ==================================================================
# DDPM training functions
# ==================================================================

def train_one_epoch_ddpm(model, ema, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for design, elevation, hand_features in loader:
        design        = design.to(device)
        elevation     = elevation.to(device)
        hand_features = hand_features.to(device)

        optimizer.zero_grad()
        loss = model(elevation, design, hand_features)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ema.update()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_ddpm(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for design, elevation, hand_features in loader:
        design        = design.to(device)
        elevation     = elevation.to(device)
        hand_features = hand_features.to(device)

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

    display_config(config)

    model_type = str(config.get('model_type', 'cvae')).lower()

    # Logger
    log_path = config.get('log_file_dir', './outputs/train.log')
    logger   = setup_logger(log_path)

    # Device
    device = get_device(config)

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

        beta_max    = float(config.get('beta_max',    4.0))
        beta_cycles = int(config.get('beta_cycles',   4))

        logger.info("=" * 60)
        logger.info(f"Training CVAE  |  fusion={config.get('fusion_method')}  "
                    f"|  val_fold={config.get('val_fold')}  "
                    f"|  epochs={total_epochs}{stop_info}")
        logger.info("=" * 60)

        for epoch in range(total_epochs):
            beta = get_cyclical_beta(epoch, total_epochs, beta_max, beta_cycles)

            t0 = time.time()
            train_loss, train_recon, train_kl = train_one_epoch_cvae(
                model, train_loader, optimizer, device, beta)
            val_loss, val_recon, val_kl = validate_cvae(
                model, val_loader, device, beta)
            scheduler.step()
            elapsed = time.time() - t0

            logger.info(
                f"Epoch {epoch+1:4d}/{total_epochs}  "
                f"beta={beta:.3f}  "
                f"train[loss={train_loss:.4f} recon={train_recon:.4f} kl={train_kl:.4f}]  "
                f"val[loss={val_loss:.4f} recon={val_recon:.4f} kl={val_kl:.4f}]  "
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
    # DDPM training
    # ============================================================
    elif model_type == 'ddpm':
        from utils.ema import EMA

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=1e-6)

        ema_decay = float(config.get('ema_decay', 0.9999))
        ema = EMA(model, decay=ema_decay)

        logger.info("=" * 60)
        logger.info(f"Training DDPM  |  T={config.get('ddpm_t', 1000)}  "
                    f"|  val_fold={config.get('val_fold')}  "
                    f"|  epochs={total_epochs}  "
                    f"|  ema_decay={ema_decay}{stop_info}")
        logger.info("=" * 60)

        for epoch in range(total_epochs):
            t0 = time.time()
            train_loss = train_one_epoch_ddpm(
                model, ema, train_loader, optimizer, device)
            val_loss = validate_ddpm(model, val_loader, device)
            scheduler.step()
            elapsed = time.time() - t0

            logger.info(
                f"Epoch {epoch+1:4d}/{total_epochs}  "
                f"train_loss={train_loss:.6f}  "
                f"val_loss={val_loss:.6f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"({elapsed:.1f}s)"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch':           epoch + 1,
                    'model_type':      'ddpm',
                    'model_state':     model.state_dict(),
                    'ema_state_dict':  ema.shadow,
                    'optimizer_state': optimizer.state_dict(),
                    'val_loss':        val_loss,
                    'config':          config,
                }, model_path)
                logger.info(f"  -> Checkpoint saved (val_loss={val_loss:.6f})")

            if early_stop_thresh > 0.0 and val_loss < early_stop_thresh:
                logger.info(
                    f"Early stop at epoch {epoch+1}: "
                    f"val_loss={val_loss:.6f} < threshold={early_stop_thresh:.4f}"
                )
                break

        logger.info("Training complete.")
        logger.info(f"Best val noise-pred loss: {best_val_loss:.6f}")

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    logger.info(f"Model saved to: {model_path}")


if __name__ == '__main__':
    main()
