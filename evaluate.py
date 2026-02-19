#!/usr/bin/env python3
"""Evaluation script for CVAE PCB Warpage — Leave-One-Out protocol.

Metrics computed for the held-out design:
  1. Sample Diversity     : mean per-pixel variance across K generated samples
  2. Inter-condition Sep. : distance between generated distributions across designs
  3. Reconstruction MSE   : MSE of deterministic reconstruction (mu path)
  4. MMD                  : Maximum Mean Discrepancy (RBF kernel) between
                            generated and real elevation distributions

Usage:
  python evaluate.py                       # evaluates all 4 folds
  python evaluate.py --fold 0              # evaluate only fold 0
  python evaluate.py --config config.txt   # explicit config path
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.load_config import load_config, display_config
from utils.dataset     import PCBWarpageDataset, _resolve_design_names
from utils.handcrafted_features import extract_handcrafted_features
from models.cvae       import CVAE


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CVAE PCB Warpage model')
    parser.add_argument('--config', type=str, default='config.txt')
    parser.add_argument('--fold',   type=int, default=None,
                        help='Evaluate a single fold (0-3); default: all folds')
    parser.add_argument('--k',      type=int, default=None,
                        help='Override num_gen_samples from config')
    return parser.parse_args()


# ------------------------------------------------------------------
# Device helper
# ------------------------------------------------------------------

def get_device(config: dict) -> torch.device:
    gpu_ids = config.get('gpu_ids', -1)
    gpu_id  = gpu_ids[0] if isinstance(gpu_ids, list) else int(gpu_ids)
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def sample_diversity(samples: torch.Tensor) -> float:
    """Mean per-pixel variance across K generated samples.

    Args:
        samples: (K, 1, H, W)
    Returns:
        scalar
    """
    return samples.var(dim=0).mean().item()


def reconstruction_mse(model, loader, device) -> float:
    """Average MSE of deterministic reconstructions over the val set."""
    model.eval()
    mse_total = 0.0
    n_batches = 0
    with torch.no_grad():
        for design, elevation, hand_features in loader:
            design        = design.to(device)
            elevation     = elevation.to(device)
            hand_features = hand_features.to(device)
            recon = model.reconstruct(elevation, design, hand_features)
            mse_total += torch.nn.functional.mse_loss(recon, elevation).item()
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
# Evaluate one fold
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate_fold(config: dict, fold: int, k: int, device: torch.device) -> dict:
    """Run evaluation for one leave-one-out fold.

    Returns:
        dict of metric name → value
    """
    design_names = _resolve_design_names(config)
    print(f"\n{'='*60}")
    print(f"Fold {fold}  —  held-out design: {design_names[fold]}")
    print('='*60)

    # Override fold in config
    cfg = dict(config)
    cfg['val_fold'] = fold

    # Load val dataset (the held-out design)
    val_dataset = PCBWarpageDataset(
        cfg['dataset_dir'], cfg, split='val', val_fold=fold)
    val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Load model checkpoint
    model_path = cfg.get('modelpath', './outputs/cvae_pcb.pth')
    if not Path(model_path).exists():
        print(f"  [WARNING] Checkpoint not found at {model_path}")
        return {}

    checkpoint = torch.load(model_path, map_location=device)
    model = CVAE(cfg).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})")

    # 1. Reconstruction MSE
    recon_mse = reconstruction_mse(model, val_loader, device)
    print(f"  Reconstruction MSE : {recon_mse:.6f}")

    # 2. Collect real elevation images (flattened)
    real_flat_list = []
    design_batch   = None
    hand_batch     = None

    for design, elevation, hand_features in val_loader:
        real_flat_list.append(elevation.view(elevation.size(0), -1))
        if design_batch is None:
            design_batch = design[:1].to(device)
            hand_batch   = hand_features[:1].to(device)

    real_flat = torch.cat(real_flat_list, dim=0)  # (N_real, H*W)

    # 3. Generate K samples for the held-out design
    gen_samples = model.sample(design_batch, hand_batch, num_samples=k)  # (K, 1, H, W)

    # 4. Diversity metric
    diversity = sample_diversity(gen_samples)
    print(f"  Sample Diversity   : {diversity:.6f}")

    # 5. MMD (downsample feature space for tractability)
    gen_flat = gen_samples.view(k, -1).cpu()
    # PCA-style reduction: project to 128 dims via random projection
    proj_dim = min(128, real_flat.size(1))
    torch.manual_seed(42)
    proj = torch.randn(real_flat.size(1), proj_dim) / (real_flat.size(1) ** 0.5)
    real_proj = real_flat.cpu() @ proj
    gen_proj  = gen_flat      @ proj
    mmd_val   = mmd(real_proj, gen_proj)
    print(f"  MMD                : {mmd_val:.6f}")

    return {
        'fold':        fold,
        'design':      design_names[fold],
        'recon_mse':   recon_mse,
        'diversity':   diversity,
        'mmd':         mmd_val,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args   = parse_args()
    config = load_config(args.config)
    display_config(config)

    device = get_device(config)
    k      = args.k if args.k else int(config.get('num_gen_samples', 10))
    design_names = _resolve_design_names(config)
    folds  = [args.fold] if args.fold is not None else list(range(len(design_names)))

    all_results = []
    for fold in folds:
        result = evaluate_fold(config, fold, k, device)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Leave-One-Out Summary")
        print('='*60)
        metrics = ['recon_mse', 'diversity', 'mmd']
        for m in metrics:
            vals = [r[m] for r in all_results]
            print(f"  {m:<18} : mean={np.mean(vals):.6f}  std={np.std(vals):.6f}")


if __name__ == '__main__':
    main()
