#!/usr/bin/env python3
"""DF²M Training Script — 3-phase training pipeline.

Phase 1: Train FNO Mean Predictor (Module A)
Phase 2: Train Residual CAE (Module B-1) using predicted means from Phase 1
Phase 3: Train OT-CFM (Module B-2) in frozen CAE latent space

Usage:
    # Full pipeline (all 3 phases)
    python dfm_approach/train.py --config dfm_approach/config_dfm.txt

    # Single phase
    python dfm_approach/train.py --config dfm_approach/config_dfm.txt --phase 1
    python dfm_approach/train.py --config dfm_approach/config_dfm.txt --phase 2
    python dfm_approach/train.py --config dfm_approach/config_dfm.txt --phase 3

    # Multi-GPU (DataParallel) — use 4 GPUs starting from GPU 0
    python dfm_approach/train.py --config dfm_approach/config_dfm.txt --num-gpus 4
    python dfm_approach/train.py --config dfm_approach/config_dfm.txt --gpu 2 --num-gpus 4

    # Resume from checkpoint
    python dfm_approach/train.py --config dfm_approach/config_dfm.txt --phase 2 --resume
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
# AMP: prefer torch.amp (PyTorch >= 2.0) with fallback for older versions
try:
    from torch.amp import GradScaler, autocast as _autocast
    def autocast_ctx(device):
        return _autocast(device_type='cuda' if device.type == 'cuda' else 'cpu')
except ImportError:
    from torch.cuda.amp import GradScaler, autocast as _autocast
    def autocast_ctx(device):
        return _autocast(enabled=device.type == 'cuda')

# Ensure dfm_approach/ is in path first (for models/, utils/)
_dfm_dir = str(Path(__file__).resolve().parent)
if _dfm_dir not in sys.path:
    sys.path.insert(0, _dfm_dir)
# Append project root so dfm_approach/utils/__init__.py can find shared modules
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.append(_project_root)
from utils.load_config import load_config, display_config
from utils.ema import EMA

from models import build_dfm_models
from utils.dfm_dataset import create_mean_dataloaders, create_residual_dataloaders
from utils.dfm_losses import fno_loss, cae_loss, get_cyclical_beta


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

class Logger:
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.f = open(log_path, 'a', encoding='utf-8')

    def log(self, msg: str):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line)
        self.f.write(line + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


# ------------------------------------------------------------------
# Multi-GPU helpers (DataParallel)
# ------------------------------------------------------------------

def _resolve_gpu_ids(gpu_start: int, num_gpus: int) -> list[int]:
    """Return list of GPU device IDs: [gpu_start, gpu_start+1, ..., gpu_start+num_gpus-1]."""
    available = torch.cuda.device_count()
    ids = [gpu_start + i for i in range(num_gpus) if (gpu_start + i) < available]
    if len(ids) < num_gpus:
        print(f"WARNING: Requested {num_gpus} GPUs from GPU {gpu_start}, "
              f"but only {available} total available. Using {len(ids)} GPUs: {ids}")
    return ids


def _wrap_dp(model: nn.Module, gpu_ids: list[int]) -> nn.Module:
    """Wrap model in DataParallel if multiple GPUs are given."""
    if len(gpu_ids) > 1:
        return nn.DataParallel(model, device_ids=gpu_ids)
    return model


def _unwrap_dp(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel to get the underlying module."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


class _CFMLossWrapper(nn.Module):
    """Thin wrapper to make OTCFM.compute_loss compatible with DataParallel.

    DataParallel only parallelizes forward(), so we route the loss computation
    through forward() and pass the frozen encoder outputs as inputs.
    """

    def __init__(self, cfm):
        super().__init__()
        self.cfm = cfm

    def forward(self, z1, c_global, c_spatial):
        return self.cfm.compute_loss(z1, c_global, c_spatial)

    # Delegate attribute access to inner cfm for optimizer, EMA, etc.
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.cfm, name)


# ------------------------------------------------------------------
# Phase 1: FNO Mean Predictor Training
# ------------------------------------------------------------------

def train_phase1(config: dict, device: torch.device, logger: Logger,
                  gpu_ids: list[int] | None = None):
    """Train Module A: FNO Mean Predictor."""
    logger.log("=" * 60)
    logger.log("PHASE 1: Training FNO Mean Predictor")
    logger.log("=" * 60)

    from models.fno_mean_predictor import FNOMeanPredictor

    model = FNOMeanPredictor(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"FNO parameters: {n_params:,}")
    if gpu_ids and len(gpu_ids) > 1:
        logger.log(f"DataParallel on GPUs: {gpu_ids}")
    model_dp = _wrap_dp(model, gpu_ids) if gpu_ids else model

    train_loader, val_loader = create_mean_dataloaders(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get('fno_lr', 0.001)),
        weight_decay=float(config.get('fno_weight_decay', 0.0001)),
    )
    epochs = int(config.get('fno_epochs', 300))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    smooth_w = float(config.get('fno_smooth_weight', 0.01))
    spectral_w = float(config.get('fno_spectral_weight', 0.1))
    early_stop = float(config.get('fno_early_stop', 0.0001))

    save_path = config.get('fno_modelpath', './outputs/dfm_fno.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_loss = float('inf')

    # FNO uses complex-valued parameters (SpectralConv2d) whose gradients are
    # incompatible with GradScaler.unscale_(). Train without AMP scaling.
    logger.log("Note: AMP GradScaler disabled for Phase 1 (complex FFT parameters)")

    for epoch in range(epochs):
        model_dp.train()
        epoch_metrics = {'mse': 0, 'smooth': 0, 'spectral': 0, 'total': 0}
        n_batches = 0

        for design, mean_warp, features in train_loader:
            design = design.to(device)
            mean_warp = mean_warp.to(device)
            features = features.to(device)

            optimizer.zero_grad()

            pred = model_dp(design, features)
            loss, metrics = fno_loss(pred, mean_warp, smooth_w, spectral_w)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k] += v
            n_batches += 1

        scheduler.step()

        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_batches, 1)

        # Validation
        model_dp.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for design, mean_warp, features in val_loader:
                design = design.to(device)
                mean_warp = mean_warp.to(device)
                features = features.to(device)
                pred = model_dp(design, features)
                val_loss += nn.functional.mse_loss(pred, mean_warp).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        logger.log(
            f"Phase1 Epoch {epoch+1}/{epochs} | "
            f"Train MSE={epoch_metrics['mse']:.6f} Smooth={epoch_metrics['smooth']:.6f} "
            f"Spec={epoch_metrics['spectral']:.6f} | Val MSE={val_loss:.6f} | "
            f"LR={scheduler.get_last_lr()[0]:.6f}"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'phase': 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': config,
                'val_loss': val_loss,
            }, save_path)
            logger.log(f"  -> Best model saved (val_loss={val_loss:.6f})")

        if early_stop > 0 and val_loss < early_stop:
            logger.log(f"  Early stopping: val_loss {val_loss:.6f} < {early_stop}")
            break

    logger.log(f"Phase 1 complete. Best val_loss={best_val_loss:.6f}")
    return model, save_path


# ------------------------------------------------------------------
# Phase 2: Residual CAE Training
# ------------------------------------------------------------------

def train_phase2(
    config: dict,
    device: torch.device,
    fno_model,
    logger: Logger,
    gpu_ids: list[int] | None = None,
):
    """Train Module B-1: Residual Conditional Autoencoder."""
    logger.log("=" * 60)
    logger.log("PHASE 2: Training Residual CAE")
    logger.log("=" * 60)

    from models.condition_encoder import ConditionEncoder
    from models.residual_cae import ResidualCAE

    cond_enc = ConditionEncoder(config).to(device)
    cae = ResidualCAE(config).to(device)

    n_params_cond = sum(p.numel() for p in cond_enc.parameters() if p.requires_grad)
    n_params_cae = sum(p.numel() for p in cae.parameters() if p.requires_grad)
    logger.log(f"ConditionEncoder params: {n_params_cond:,}")
    logger.log(f"ResidualCAE params: {n_params_cae:,}")
    if gpu_ids and len(gpu_ids) > 1:
        logger.log(f"DataParallel on GPUs: {gpu_ids}")
    cond_enc_dp = _wrap_dp(cond_enc, gpu_ids) if gpu_ids else cond_enc
    cae_dp = _wrap_dp(cae, gpu_ids) if gpu_ids else cae

    # Create residual dataset using trained FNO
    fno_model.eval()
    train_loader, val_loader = create_residual_dataloaders(config, fno_model, device)

    params = list(cond_enc.parameters()) + list(cae.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config.get('cae_lr', 0.0001)),
        weight_decay=float(config.get('cae_weight_decay', 0.0001)),
    )
    epochs = int(config.get('cae_epochs', 200))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    beta_max = float(config.get('cae_beta_max', 0.3))
    beta_cycles = int(config.get('cae_beta_cycles', 6))
    free_bits = float(config.get('cae_free_bits', 0.5))
    spectral_w = float(config.get('cae_spectral_weight', 0.1))

    save_path = config.get('cae_modelpath', './outputs/dfm_cae.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        cond_enc_dp.train()
        cae_dp.train()

        beta = get_cyclical_beta(epoch, epochs, beta_max, beta_cycles, warmup_epochs=10)

        epoch_metrics = {
            'recon_mse': 0, 'kl_raw': 0, 'spectral': 0,
            'total': 0, 'active_kl_dims': 0,
        }
        n_batches = 0

        for residual, design, features in train_loader:
            residual = residual.to(device)
            design = design.to(device)
            features = features.to(device)

            optimizer.zero_grad()

            with autocast_ctx(device):
                c_global, c_spatial = cond_enc_dp(design, features)
                recon, mu, logvar = cae_dp(residual, c_global, c_spatial)
                loss, metrics = cae_loss(
                    recon, residual, mu, logvar,
                    beta=beta, free_bits=free_bits, spectral_weight=spectral_w,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            for k in epoch_metrics:
                if k in metrics:
                    epoch_metrics[k] += metrics[k]
            n_batches += 1

        scheduler.step()
        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_batches, 1)

        # Validation
        cond_enc_dp.eval()
        cae_dp.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for residual, design, features in val_loader:
                residual = residual.to(device)
                design = design.to(device)
                features = features.to(device)
                c_global, c_spatial = cond_enc_dp(design, features)
                recon, mu, logvar = cae_dp(residual, c_global, c_spatial)
                val_loss += nn.functional.mse_loss(recon, residual).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        logger.log(
            f"Phase2 Epoch {epoch+1}/{epochs} | "
            f"Recon={epoch_metrics['recon_mse']:.6f} KL={epoch_metrics['kl_raw']:.4f} "
            f"ActiveDims={epoch_metrics['active_kl_dims']:.0f} β={beta:.4f} | "
            f"Val Recon={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'phase': 2,
                'cond_enc_state': cond_enc.state_dict(),
                'cae_state': cae.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': config,
                'val_loss': val_loss,
            }, save_path)
            logger.log(f"  -> Best model saved (val_loss={val_loss:.6f})")

    logger.log(f"Phase 2 complete. Best val_loss={best_val_loss:.6f}")
    return cond_enc, cae, save_path


# ------------------------------------------------------------------
# Phase 3: OT-CFM Training
# ------------------------------------------------------------------

def train_phase3(
    config: dict,
    device: torch.device,
    fno_model,
    cond_enc: nn.Module,
    cae: nn.Module,
    logger: Logger,
    gpu_ids: list[int] | None = None,
):
    """Train Module B-2: OT-Conditional Flow Matching."""
    logger.log("=" * 60)
    logger.log("PHASE 3: Training OT-CFM Velocity Network")
    logger.log("=" * 60)

    from models.ot_cfm import OTCFM

    cfm = OTCFM(config).to(device)
    n_params = sum(p.numel() for p in cfm.parameters() if p.requires_grad)
    logger.log(f"OT-CFM velocity net params: {n_params:,}")
    if gpu_ids and len(gpu_ids) > 1:
        logger.log(f"DataParallel on GPUs: {gpu_ids}")

    # Freeze FNO, cond_enc, CAE encoder
    fno_model.eval()
    cond_enc.eval()
    cae.eval()
    for p in fno_model.parameters():
        p.requires_grad_(False)
    for p in cond_enc.parameters():
        p.requires_grad_(False)
    for p in cae.parameters():
        p.requires_grad_(False)

    # Wrap CFM loss in DataParallel-compatible module
    # DataParallel only parallelizes forward(), so _CFMLossWrapper routes
    # compute_loss through forward()
    cfm_wrapper = _CFMLossWrapper(cfm)
    cfm_dp = _wrap_dp(cfm_wrapper, gpu_ids) if gpu_ids else cfm_wrapper

    # Residual dataset
    train_loader, val_loader = create_residual_dataloaders(config, fno_model, device)

    optimizer = torch.optim.AdamW(
        cfm.parameters(),
        lr=float(config.get('cfm_lr', 0.0002)),
        weight_decay=float(config.get('cfm_weight_decay', 0.0001)),
    )
    epochs = int(config.get('cfm_epochs', 500))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    ema_decay = float(config.get('cfm_ema_decay', 0.9999))
    ema = EMA(cfm, decay=ema_decay)

    save_path = config.get('cfm_modelpath', './outputs/dfm_cfm.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        cfm_dp.train()
        epoch_loss = 0
        n_batches = 0

        for residual, design, features in train_loader:
            residual = residual.to(device)
            design = design.to(device)
            features = features.to(device)

            # Encode condition (frozen, no DP needed)
            with torch.no_grad():
                c_global, c_spatial = cond_enc(design, features)
                mu, logvar = cae.encode(residual)
                z1 = mu  # Use mean (deterministic) for flow target

            optimizer.zero_grad()

            with autocast_ctx(device):
                # cfm_dp.forward(z1, c_global, c_spatial) → compute_loss
                loss = cfm_dp(z1, c_global, c_spatial)
                if loss.dim() > 0:
                    loss = loss.mean()  # DataParallel returns per-GPU losses

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(cfm.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss /= max(n_batches, 1)

        # Validation (single GPU is sufficient)
        cfm.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for residual, design, features in val_loader:
                residual = residual.to(device)
                design = design.to(device)
                features = features.to(device)
                c_global, c_spatial = cond_enc(design, features)
                mu, _ = cae.encode(residual)
                loss = cfm.compute_loss(mu, c_global, c_spatial)
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(n_val, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.log(
                f"Phase3 Epoch {epoch+1}/{epochs} | "
                f"Train VelMSE={epoch_loss:.6f} | Val VelMSE={val_loss:.6f} | "
                f"LR={scheduler.get_last_lr()[0]:.6f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'phase': 3,
                'cfm_state': cfm.state_dict(),
                'ema_state': ema.shadow,
                'optimizer_state': optimizer.state_dict(),
                'config': config,
                'val_loss': val_loss,
                'fno_checkpoint': config.get('fno_modelpath'),
                'cae_checkpoint': config.get('cae_modelpath'),
            }, save_path)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.log(f"  -> Best model saved (val_loss={val_loss:.6f})")

    logger.log(f"Phase 3 complete. Best val_loss={best_val_loss:.6f}")
    return cfm, ema, save_path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='DF²M Training Pipeline')
    parser.add_argument('--config', type=str, default='dfm_approach/config_dfm.txt')
    parser.add_argument('--phase', type=int, default=0,
                        help='Phase to run: 0=all, 1=FNO, 2=CAE, 3=CFM')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing checkpoints')
    parser.add_argument('--val_fold', type=int, default=None,
                        help='Override val_fold from config')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device index (overrides config gpu_ids)')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs for DataParallel (default: 1)')
    parser.add_argument('--tag', type=str, default=None,
                        help='Tag for checkpoint/log naming (e.g., fold0)')
    args = parser.parse_args()

    config = load_config(args.config)
    display_config(config)

    if args.val_fold is not None:
        config['val_fold'] = args.val_fold

    # Tag-based output paths: allow per-fold checkpoint naming
    if args.tag:
        tag = args.tag
        for key in ('fno_modelpath', 'cae_modelpath', 'cfm_modelpath'):
            path = config.get(key, '')
            if path:
                base, ext = os.path.splitext(path)
                config[key] = f"{base}_{tag}{ext}"
        log_dir = config.get('log_file_dir', './outputs/train_dfm.log')
        base, ext = os.path.splitext(log_dir)
        config['log_file_dir'] = f"{base}_{tag}{ext}"

    # Device
    if args.gpu is not None:
        gpu_start = args.gpu
    else:
        cfg_gpu = config.get('gpu_ids', 0)
        gpu_start = cfg_gpu[0] if isinstance(cfg_gpu, list) else cfg_gpu

    num_gpus = args.num_gpus
    if gpu_start >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_start}')
        torch.cuda.set_device(device)
        dp_gpu_ids = _resolve_gpu_ids(gpu_start, num_gpus) if num_gpus > 1 else None
    else:
        device = torch.device('cpu')
        dp_gpu_ids = None

    log_path = config.get('log_file_dir', './outputs/train_dfm.log')
    logger = Logger(log_path)
    logger.log(f"Device: {device}" + (f" (DataParallel: {dp_gpu_ids})" if dp_gpu_ids else ""))
    logger.log(f"Config: {args.config}")

    run_phases = [1, 2, 3] if args.phase == 0 else [args.phase]

    # ------ Phase 1 ------
    fno_model = None
    if 1 in run_phases:
        fno_model, fno_path = train_phase1(config, device, logger, gpu_ids=dp_gpu_ids)
    else:
        # Load pretrained FNO
        fno_path = config.get('fno_modelpath', './outputs/dfm_fno.pth')
        if os.path.exists(fno_path):
            from models.fno_mean_predictor import FNOMeanPredictor
            fno_model = FNOMeanPredictor(config).to(device)
            ckpt = torch.load(fno_path, map_location=device, weights_only=False)
            fno_model.load_state_dict(ckpt['model_state'])
            logger.log(f"Loaded FNO from {fno_path}")
        else:
            logger.log(f"[ERROR] FNO checkpoint not found: {fno_path}")
            sys.exit(1)

    # ------ Phase 2 ------
    cond_enc = None
    cae = None
    if 2 in run_phases:
        cond_enc, cae, cae_path = train_phase2(config, device, fno_model, logger, gpu_ids=dp_gpu_ids)
    elif 3 in run_phases:
        # Load pretrained CAE
        cae_path = config.get('cae_modelpath', './outputs/dfm_cae.pth')
        if os.path.exists(cae_path):
            from models.condition_encoder import ConditionEncoder
            from models.residual_cae import ResidualCAE
            cond_enc = ConditionEncoder(config).to(device)
            cae = ResidualCAE(config).to(device)
            ckpt = torch.load(cae_path, map_location=device, weights_only=False)
            cond_enc.load_state_dict(ckpt['cond_enc_state'])
            cae.load_state_dict(ckpt['cae_state'])
            logger.log(f"Loaded CAE from {cae_path}")
        else:
            logger.log(f"[ERROR] CAE checkpoint not found: {cae_path}")
            sys.exit(1)

    # ------ Phase 3 ------
    if 3 in run_phases:
        train_phase3(config, device, fno_model, cond_enc, cae, logger, gpu_ids=dp_gpu_ids)

    logger.log("Training pipeline complete.")
    logger.close()


if __name__ == '__main__':
    main()
