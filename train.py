#!/usr/bin/env python3
"""Training script for CVAE PCB Warpage.

Usage:
  python train.py                      # uses config.txt in current dir
  python train.py --config config.txt  # explicit path
  python train.py --val_fold 1         # override leave-one-out fold

Training loop:
  - Cyclical KL annealing (beta)
  - Checkpoint saved each epoch if validation loss improves
  - Log written to log_file_dir
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

from utils.load_config import load_config, display_config
from utils.dataset     import create_dataloaders
from utils.losses      import cvae_loss, get_cyclical_beta
from models.cvae       import CVAE


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Train CVAE PCB Warpage model')
    parser.add_argument('--config',   type=str, default='config.txt')
    parser.add_argument('--val_fold', type=int, default=None,
                        help='Override val_fold from config (0-3)')
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
    logger = logging.getLogger('CVAE_PCB')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ------------------------------------------------------------------
# Training epoch
# ------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, beta, epoch):
    model.train()
    total_loss = recon_sum = kl_sum = 0.0
    n_batches  = 0

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


# ------------------------------------------------------------------
# Validation epoch
# ------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, beta):
    model.eval()
    total_loss = recon_sum = kl_sum = 0.0
    n_batches  = 0

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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args   = parse_args()
    config = load_config(args.config)

    if args.val_fold is not None:
        config['val_fold'] = args.val_fold

    display_config(config)

    # Logger
    log_path = config.get('log_file_dir', './outputs/train.log')
    logger   = setup_logger(log_path)

    # Device
    device = get_device(config)

    # Data
    train_loader, val_loader = create_dataloaders(config)

    # Model
    model = CVAE(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,}")

    # Optimizer
    lr           = float(config.get('learning_rate', 1e-4))
    weight_decay = float(config.get('weight_decay',  1e-4))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # LR scheduler: cosine annealing over full training
    total_epochs = int(config.get('training_epochs', 200))
    scheduler    = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    # KL annealing params
    beta_max    = float(config.get('beta_max',    4.0))
    beta_cycles = int(config.get('beta_cycles',   4))

    # Checkpoint
    model_path = config.get('modelpath', './outputs/cvae_pcb.pth')
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')

    logger.info("=" * 60)
    logger.info(f"Training CVAE  |  fusion={config.get('fusion_method')}  "
                f"|  val_fold={config.get('val_fold')}  "
                f"|  epochs={total_epochs}")
    logger.info("=" * 60)

    for epoch in range(total_epochs):
        beta = get_cyclical_beta(epoch, total_epochs, beta_max, beta_cycles)

        t0 = time.time()
        train_loss, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, device, beta, epoch)
        val_loss, val_recon, val_kl = validate(
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

        # Save best checkpoint (based on val reconstruction loss — KL can spike)
        if val_recon < best_val_loss:
            best_val_loss = val_recon
            torch.save({
                'epoch':      epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss':   val_loss,
                'val_recon':  val_recon,
                'config':     config,
            }, model_path)
            logger.info(f"  → Checkpoint saved (val_recon={val_recon:.4f})")

    logger.info("Training complete.")
    logger.info(f"Best val recon loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {model_path}")


if __name__ == '__main__':
    main()
