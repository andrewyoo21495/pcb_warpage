#!/usr/bin/env python3
"""Evaluation script for PCB Warpage models — Leave-One-Out protocol.

Supports both CVAE and DDPM (auto-detected from checkpoint).

Two-step validation:
  Step 1 — Memorisation check (train split)
    - Train Recon MSE : can the model reconstruct designs it has seen?
    - Real Diversity  : how much do the real elevation samples vary? (baseline)

  Step 2 — Generalisation check (val / held-out split)
    - Val Recon MSE   : reconstruction on the held-out design (CVAE only)
    - Active KL dims  : are latent dims being used? (CVAE only)
    - Gen Diversity   : per-pixel variance across K generated samples
    - MMD             : distance between generated and real distributions

Usage:
  python evaluate.py                       # evaluates all folds (both steps)
  python evaluate.py --fold 0              # evaluate only fold 0
  python evaluate.py --step 1              # memorisation check only
  python evaluate.py --step 2              # generalisation check only
  python evaluate.py --config config.txt   # explicit config path
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.load_config import load_config, display_config
from utils.dataset     import PCBWarpageDataset, _resolve_design_names
from models            import build_model


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PCB Warpage model')
    parser.add_argument('--config', type=str, default='config.txt')
    parser.add_argument('--fold',   type=int, default=None,
                        help='Evaluate a single fold (0-indexed); default: all folds')
    parser.add_argument('--k',      type=int, default=None,
                        help='Override num_gen_samples from config')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Max samples to generate per GPU batch (reduces VRAM usage)')
    parser.add_argument('--cpu',    action='store_true',
                        help='Force CPU evaluation (avoids GPU usage entirely)')
    parser.add_argument('--step',   type=int, default=None, choices=[1, 2],
                        help='Run only step 1 (memorisation) or step 2 (generalisation)')
    return parser.parse_args()


# ------------------------------------------------------------------
# Device helper
# ------------------------------------------------------------------

def get_device(config: dict) -> torch.device:
    gpu_ids = config.get('gpu_ids', -1)
    gpu_id  = gpu_ids[0] if isinstance(gpu_ids, list) else int(gpu_ids)
    if gpu_id >= 0 and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

@torch.no_grad()
def active_kl_dims(model, loader, device, use_amp: bool,
                   threshold: float = 0.1) -> tuple[int, int]:
    """Count latent dimensions whose mean KL exceeds `threshold` nats.

    Runs the elevation encoder over the dataset and computes the per-dim KL
    averaged across all batches.  A dimension is 'active' (not collapsed) when
    its mean KL is above the threshold — meaning the encoder learned to encode
    real information there rather than defaulting to the prior N(0, 1).

    Args:
        model     : CVAE model (must have a .forward() returning (recon, mu, logvar))
        loader    : DataLoader over the evaluation set
        device    : torch.device
        use_amp   : enable automatic mixed precision
        threshold : minimum mean KL (nats) for a dim to be considered active

    Returns:
        (n_active, z_dim)
    """
    model.eval()
    kl_per_dim_acc = None
    n_batches = 0

    for design, elevation, hand_features in loader:
        design        = design.to(device, non_blocking=True)
        elevation     = elevation.to(device, non_blocking=True)
        hand_features = hand_features.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            _, mu, logvar = model(elevation, design, hand_features)
        mu, logvar = mu.float(), logvar.float()
        kl_dims = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=0)
        kl_per_dim_acc = kl_dims if kl_per_dim_acc is None else kl_per_dim_acc + kl_dims
        n_batches += 1

    if kl_per_dim_acc is None:
        return 0, model.z_dim

    mean_kl = kl_per_dim_acc / max(n_batches, 1)
    n_active = int((mean_kl > threshold).sum().item())
    return n_active, model.z_dim


def real_diversity(samples: torch.Tensor) -> float:
    """Mean per-pixel variance across real elevation samples.

    Args:
        samples: (N, D) flattened real elevations, or (N, 1, H, W)
    Returns:
        scalar
    """
    if samples.dim() == 4:
        samples = samples.view(samples.size(0), -1)
    return samples.float().var(dim=0).mean().item()


def sample_diversity(samples: torch.Tensor) -> float:
    """Mean per-pixel variance across K generated samples.

    Args:
        samples: (K, 1, H, W)
    Returns:
        scalar
    """
    return samples.var(dim=0).mean().item()


def reconstruction_mse(model, loader, device, use_amp: bool) -> float:
    """Average MSE of deterministic reconstructions (CVAE only)."""
    model.eval()
    mse_total = 0.0
    n_batches = 0
    with torch.no_grad():
        for design, elevation, hand_features in loader:
            design        = design.to(device, non_blocking=True)
            elevation     = elevation.to(device, non_blocking=True)
            hand_features = hand_features.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                recon = model.reconstruct(elevation, design, hand_features)
            mse_total += torch.nn.functional.mse_loss(recon.float(), elevation).item()
            n_batches += 1
    return mse_total / max(n_batches, 1)


def _rbf_kernel(X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """RBF (Gaussian) kernel between rows of X and Y."""
    XX = (X ** 2).sum(1, keepdim=True)
    YY = (Y ** 2).sum(1, keepdim=True)
    dist = XX + YY.t() - 2 * X @ Y.t()
    return torch.exp(-dist / (2 * sigma ** 2))


def mmd(real: torch.Tensor, generated: torch.Tensor, sigma: float = 1.0) -> float:
    """Unbiased MMD^2 estimate with RBF kernel.

    Args:
        real      : (N, D)  flattened real samples
        generated : (M, D)  flattened generated samples
    """
    K_rr = _rbf_kernel(real,      real,      sigma).mean()
    K_gg = _rbf_kernel(generated, generated, sigma).mean()
    K_rg = _rbf_kernel(real,      generated, sigma).mean()
    return (K_rr + K_gg - 2 * K_rg).item()


# ------------------------------------------------------------------
# Model loading helper
# ------------------------------------------------------------------

def load_model_from_checkpoint(checkpoint: dict, config: dict, device: torch.device):
    """Load a model from checkpoint, handling both CVAE and DDPM.

    For DDPM checkpoints, EMA weights are loaded for inference.
    """
    model_type = checkpoint.get('model_type', 'cvae')
    # Override config model_type so build_model creates the right class
    config['model_type'] = model_type

    model = build_model(config).to(device)

    if model_type == 'ddpm' and 'ema_state_dict' in checkpoint:
        # Load EMA weights for inference (better generation quality)
        ema_sd = checkpoint['ema_state_dict']
        model_sd = model.state_dict()
        for name in ema_sd:
            if name in model_sd:
                model_sd[name] = ema_sd[name]
        model.load_state_dict(model_sd)
        print(f"  Loaded DDPM checkpoint with EMA weights")
    else:
        model.load_state_dict(checkpoint['model_state'])

    model.eval()
    return model, model_type


# ------------------------------------------------------------------
# Evaluate one fold
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate_fold(config: dict, fold: int, k: int, device: torch.device,
                  batch_size: int = None, step: int = None) -> dict:
    """Run evaluation for one leave-one-out fold.

    Args:
        batch_size : Max samples per forward pass. If None, generate all K at once.
        step       : 1 = memorisation check only, 2 = generalisation check only,
                     None = both steps.

    Returns:
        dict of metric name -> value
    """
    design_names = _resolve_design_names(config)
    print(f"\n{'='*60}")
    print(f"Fold {fold}  --  held-out design: {design_names[fold]}")
    print('='*60)

    cfg = dict(config)
    cfg['val_fold'] = fold
    use_amp = device.type == 'cuda'

    # Load model checkpoint
    model_path = cfg.get('modelpath', './outputs/cvae_pcb.pth')
    if not Path(model_path).exists():
        print(f"  [WARNING] Checkpoint not found at {model_path}")
        return {}

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model, model_type = load_model_from_checkpoint(checkpoint, cfg, device)
    print(f"Loaded {model_type.upper()} checkpoint (epoch {checkpoint.get('epoch', '?')})")

    result = {'fold': fold, 'design': design_names[fold]}

    # ==================================================================
    # Step 1 — Memorisation check (train split)
    # ==================================================================
    run_step1 = step in (None, 1)
    train_recon_mse = float('nan')

    if run_step1:
        print(f"\n[Step 1] Memorisation check — train split")

        train_dataset = PCBWarpageDataset(
            cfg['dataset_dir'], cfg, split='train', val_fold=fold)
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=False, num_workers=0)

        if model_type == 'cvae':
            train_recon_mse = reconstruction_mse(model, train_loader, device, use_amp)
            verdict = 'good' if train_recon_mse < 0.005 else 'high — model may be underfitting'
            print(f"  Train Recon MSE  : {train_recon_mse:.6f}  ({verdict})")
        else:
            print(f"  Train Recon MSE  : N/A (DDPM)")

        # Real sample diversity — baseline for comparison with generated diversity
        real_train_list = []
        for _, elevation, _ in train_loader:
            real_train_list.append(elevation.view(elevation.size(0), -1))
        real_train_flat = torch.cat(real_train_list, dim=0)
        real_div = real_diversity(real_train_flat)
        print(f"  Real Diversity   : {real_div:.6f}  "
              f"(baseline — generated diversity should be in this ballpark)")

        result['train_recon_mse'] = train_recon_mse
        result['real_diversity']  = real_div

    # ==================================================================
    # Step 2 — Generalisation check (val / held-out split)
    # ==================================================================
    run_step2 = step in (None, 2)
    val_recon_mse = float('nan')
    n_active      = None
    gen_diversity = float('nan')
    mmd_val       = float('nan')

    if run_step2:
        print(f"\n[Step 2] Generalisation check — val split (held-out: {design_names[fold]})")

        val_dataset = PCBWarpageDataset(
            cfg['dataset_dir'], cfg, split='val', val_fold=fold)
        val_loader  = DataLoader(
            val_dataset, batch_size=32, shuffle=False, num_workers=0)

        # 2a. Val reconstruction MSE + active KL dims (CVAE only)
        if model_type == 'cvae':
            val_recon_mse = reconstruction_mse(model, val_loader, device, use_amp)
            verdict = 'good' if val_recon_mse < 0.01 else 'high'
            print(f"  Val Recon MSE    : {val_recon_mse:.6f}  ({verdict})")

            n_active, z_dim = active_kl_dims(model, val_loader, device, use_amp)
            status = 'healthy' if n_active > z_dim // 4 else 'WARNING: possible posterior collapse'
            print(f"  Active KL dims   : {n_active}/{z_dim}  ({status})")
        else:
            print(f"  Val Recon MSE    : N/A (DDPM)")

        # 2b. Collect real val elevations
        real_flat_list = []
        design_batch   = None
        hand_batch     = None

        for design, elevation, hand_features in val_loader:
            real_flat_list.append(elevation.view(elevation.size(0), -1))
            if design_batch is None:
                design_batch = design[:1].to(device, non_blocking=True)
                hand_batch   = hand_features[:1].to(device, non_blocking=True)

        real_flat = torch.cat(real_flat_list, dim=0)  # (N_real, H*W)

        # 2c. Generate K samples (batched to limit VRAM)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if batch_size and batch_size < k:
                chunks = []
                remaining = k
                while remaining > 0:
                    n = min(batch_size, remaining)
                    chunks.append(model.sample(design_batch, hand_batch, num_samples=n))
                    remaining -= n
                gen_samples = torch.cat(chunks, dim=0)
            else:
                gen_samples = model.sample(design_batch, hand_batch, num_samples=k)
        gen_samples = gen_samples.float()

        # 2d. Generated diversity
        gen_diversity = sample_diversity(gen_samples)
        real_val_div  = real_diversity(real_flat)
        ratio = gen_diversity / real_val_div if real_val_div > 0 else float('nan')
        print(f"  Real Diversity   : {real_val_div:.6f}  (val split baseline)")
        print(f"  Gen  Diversity   : {gen_diversity:.6f}  "
              f"(ratio vs real: {ratio:.2f}x  "
              f"{'good' if 0.3 <= ratio <= 3.0 else 'low — try higher temperature'})")

        # 2e. MMD
        gen_flat = gen_samples.view(k, -1).cpu()
        proj_dim = min(128, real_flat.size(1))
        torch.manual_seed(42)
        proj = torch.randn(real_flat.size(1), proj_dim) / (real_flat.size(1) ** 0.5)
        real_proj = real_flat.cpu() @ proj
        gen_proj  = gen_flat        @ proj
        mmd_val   = mmd(real_proj, gen_proj)
        print(f"  MMD              : {mmd_val:.6f}  "
              f"({'good' if mmd_val < 0.1 else 'moderate' if mmd_val < 0.3 else 'high — distributions differ'})")

        result.update({
            'val_recon_mse': val_recon_mse,
            'active_dims':   n_active,
            'gen_diversity': gen_diversity,
            'mmd':           mmd_val,
        })

    return result


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args   = parse_args()
    config = load_config(args.config)
    display_config(config)

    device = torch.device('cpu') if args.cpu else get_device(config)
    k      = args.k if args.k else int(config.get('num_gen_samples', 10))
    batch_size = args.batch_size
    design_names = _resolve_design_names(config)
    folds  = [args.fold] if args.fold is not None else list(range(len(design_names)))

    all_results = []
    for fold in folds:
        result = evaluate_fold(config, fold, k, device,
                               batch_size=batch_size, step=args.step)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Leave-One-Out Summary")
        print('='*60)

        step = args.step

        # Step 1 summary
        if step in (None, 1):
            print("[Step 1 — Memorisation]")
            for m, label in [('train_recon_mse', 'train_recon_mse'),
                              ('real_diversity',  'real_diversity')]:
                vals = [r[m] for r in all_results
                        if m in r and not np.isnan(r[m])]
                if vals:
                    print(f"  {label:<20} : mean={np.mean(vals):.6f}  std={np.std(vals):.6f}")
                else:
                    print(f"  {label:<20} : N/A")

        # Step 2 summary
        if step in (None, 2):
            print("[Step 2 — Generalisation]")
            for m, label in [('val_recon_mse', 'val_recon_mse'),
                              ('gen_diversity', 'gen_diversity'),
                              ('mmd',           'mmd')]:
                vals = [r[m] for r in all_results
                        if m in r and not np.isnan(r[m])]
                if vals:
                    print(f"  {label:<20} : mean={np.mean(vals):.6f}  std={np.std(vals):.6f}")
                else:
                    print(f"  {label:<20} : N/A")
            active_vals = [r['active_dims'] for r in all_results
                           if r.get('active_dims') is not None]
            if active_vals:
                print(f"  {'active_kl_dims':<20} : mean={np.mean(active_vals):.1f}  "
                      f"min={min(active_vals)}  max={max(active_vals)}")


if __name__ == '__main__':
    main()
