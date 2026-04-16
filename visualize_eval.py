#!/usr/bin/env python3
"""Visual quality check for trained CVAE/DDPM models.

For each held-out design (leave-one-out fold), generates four plots that
compare the model's generated samples against the real elevation data:

  Panel A — Mean & Std images
    [Design | Real Mean | Real Std | Gen Mean | Gen Std | Diff (mean)]
    Shows whether the model captures the right spatial structure on average
    and whether its spread matches the real sample spread.

  Panel B — Sample grid
    [N real samples] on top, [N generated samples] on bottom.
    Direct visual comparison of individual texture and structure.

  Panel C — Pixel value histogram
    Overlaid density histogram of real vs generated pixel values.
    Checks whether the overall intensity distribution matches.

  Panel D — Fold summary bar chart (multi-fold runs only)
    Gen/Real diversity ratio and MMD per fold, so you can spot which
    designs generalise well and which do not.

Usage:
  python visualize_eval.py                        # all folds, k from config
  python visualize_eval.py --fold 0               # single fold
  python visualize_eval.py --k 32                 # override num samples
  python visualize_eval.py --save outputs/vis     # output directory
  python visualize_eval.py --show                 # also open windows
  python visualize_eval.py --grid-n 8             # samples per row in grid
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from utils.load_config import load_config, display_config
from utils.dataset     import PCBWarpageDataset, _resolve_design_names
from models            import build_model


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Visualise CVAE/DDPM generation quality')
    parser.add_argument('--config',   type=str, default='config.txt')
    parser.add_argument('--fold',     type=int, default=None,
                        help='Evaluate a single fold (0-indexed); default: all folds')
    parser.add_argument('--k',        type=int, default=None,
                        help='Number of samples to generate per design (overrides config)')
    parser.add_argument('--save',     type=str, default='outputs/vis',
                        help='Directory to save visualisation PNGs')
    parser.add_argument('--show',     action='store_true',
                        help='Also display plots interactively')
    parser.add_argument('--grid-n',   type=int, default=8,
                        help='Number of samples to show per row in the sample grid (default: 8)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0)')
    parser.add_argument('--cpu',      action='store_true',
                        help='Force CPU')
    return parser.parse_args()


# ------------------------------------------------------------------
# Device / model helpers  (same pattern as evaluate.py)
# ------------------------------------------------------------------

def get_device(config, cpu=False):
    if cpu:
        return torch.device('cpu')
    gpu_ids = config.get('gpu_ids', -1)
    gpu_id  = gpu_ids[0] if isinstance(gpu_ids, list) else int(gpu_ids)
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def load_model(config, device):
    model_path = config.get('modelpath', './outputs/cvae_pcb.pth')
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}  (run train.py first)")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_type = ckpt.get('model_type', 'cvae')
    config['model_type'] = model_type

    model = build_model(config).to(device)

    if model_type == 'ddpm' and 'ema_state_dict' in ckpt:
        sd = model.state_dict()
        for k, v in ckpt['ema_state_dict'].items():
            if k in sd:
                sd[k] = v
        model.load_state_dict(sd)
        print(f"Loaded DDPM (EMA weights) from {model_path}")
    else:
        model.load_state_dict(ckpt['model_state'])
        print(f"Loaded {model_type.upper()} from {model_path}  "
              f"(epoch {ckpt.get('epoch', '?')})")

    model.eval()
    return model, model_type


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------

@torch.no_grad()
def collect_real_samples(config, fold, device):
    """Return (real_tensor, design_tensor, hand_tensor) for the val fold.

    real_tensor  : (N, 1, H, W)  all real elevation images in [0,1]
    design_tensor: (1, 1, H, W)  the design image (first sample)
    hand_tensor  : (1, HAND_DIM) handcrafted features (first sample)
    """
    cfg = dict(config)
    cfg['val_fold'] = fold

    dataset = PCBWarpageDataset(cfg['dataset_dir'], cfg, split='val', val_fold=fold)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    real_list    = []
    design_batch = None
    hand_batch   = None

    for design, elevation, hand in loader:
        real_list.append(elevation)
        if design_batch is None:
            design_batch = design[:1].to(device)
            hand_batch   = hand[:1].to(device)

    real_tensor = torch.cat(real_list, dim=0)   # (N, 1, H, W)  stays on CPU
    return real_tensor, design_batch, hand_batch


@torch.no_grad()
def generate_samples(model, design_batch, hand_batch, k, temperature):
    """Generate k samples; returns (K, 1, H, W) on CPU."""
    samples = model.sample(design_batch, hand_batch,
                           num_samples=k, temperature=temperature)
    return samples.cpu()


# ------------------------------------------------------------------
# Panel A — Mean & Std images
# ------------------------------------------------------------------

def plot_mean_std(design_np, real_tensor, gen_tensor, design_name, save_path, show):
    """
    Row layout:
      Design | Real Mean | Real Std | Gen Mean | Gen Std | |Mean diff|
    """
    real = real_tensor.squeeze(1).numpy()   # (N, H, W)
    gen  = gen_tensor.squeeze(1).numpy()    # (K, H, W)

    real_mean = real.mean(axis=0)
    real_std  = real.std(axis=0)
    gen_mean  = gen.mean(axis=0)
    gen_std   = gen.std(axis=0)
    diff      = np.abs(gen_mean - real_mean)

    vmin, vmax = 0.0, 1.0
    std_max = max(real_std.max(), gen_std.max(), 1e-6)

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    fig.suptitle(f"Mean & Std comparison — {design_name}", fontsize=13)

    panels = [
        (design_np,  'Design',           'gray',    vmin,    vmax),
        (real_mean,  f'Real Mean (n={real.shape[0]})', 'gray', vmin, vmax),
        (real_std,   'Real Std',         'hot',     0,       std_max),
        (gen_mean,   f'Gen Mean (k={gen.shape[0]})',  'gray', vmin, vmax),
        (gen_std,    'Gen Std',          'hot',     0,       std_max),
        (diff,       '|Mean Diff|',      'RdYlGn_r', 0,      diff.max() + 1e-6),
    ]

    for ax, (img, title, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate stats
    axes[5].set_xlabel(
        f"mean={diff.mean():.4f}  max={diff.max():.4f}", fontsize=8)

    fig.tight_layout()
    _save_and_show(fig, save_path, show)


# ------------------------------------------------------------------
# Panel B — Sample grid
# ------------------------------------------------------------------

def plot_sample_grid(real_tensor, gen_tensor, design_name, save_path, show,
                     n_per_row=8):
    """
    Top rows   : up to n_per_row real samples
    Bottom rows: up to n_per_row generated samples
    """
    real = real_tensor.squeeze(1).numpy()
    gen  = gen_tensor.squeeze(1).numpy()

    n_real = min(n_per_row, real.shape[0])
    n_gen  = min(n_per_row, gen.shape[0])
    n_cols = max(n_real, n_gen)

    fig, axes = plt.subplots(2, n_cols, figsize=(2.2 * n_cols, 5))
    fig.suptitle(f"Sample grid — {design_name}", fontsize=13)

    for col in range(n_cols):
        # Real row
        ax_r = axes[0, col]
        if col < n_real:
            ax_r.imshow(real[col], cmap='gray', vmin=0, vmax=1)
            ax_r.set_title(f'Real {col+1}', fontsize=7)
        else:
            ax_r.axis('off')
        ax_r.axis('off')

        # Gen row
        ax_g = axes[1, col]
        if col < n_gen:
            ax_g.imshow(gen[col], cmap='gray', vmin=0, vmax=1)
            ax_g.set_title(f'Gen {col+1}', fontsize=7)
        else:
            ax_g.axis('off')
        ax_g.axis('off')

    # Row labels
    axes[0, 0].set_ylabel('Real', fontsize=10)
    axes[1, 0].set_ylabel('Generated', fontsize=10)
    for ax in axes[:, 0]:
        ax.axis('off')

    fig.tight_layout()
    _save_and_show(fig, save_path, show)


# ------------------------------------------------------------------
# Panel C — Pixel histogram
# ------------------------------------------------------------------

def plot_histogram(real_tensor, gen_tensor, design_name, save_path, show):
    """Overlaid pixel value density histogram, real vs generated."""
    real_px = real_tensor.numpy().ravel()
    gen_px  = gen_tensor.numpy().ravel()

    real_mean, real_std = real_px.mean(), real_px.std()
    gen_mean,  gen_std  = gen_px.mean(),  gen_px.std()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(real_px, bins=100, range=(0, 1), density=True,
            alpha=0.55, color='steelblue', label=f'Real  μ={real_mean:.3f} σ={real_std:.3f}')
    ax.hist(gen_px,  bins=100, range=(0, 1), density=True,
            alpha=0.55, color='tomato',    label=f'Gen   μ={gen_mean:.3f} σ={gen_std:.3f}')
    ax.axvline(real_mean, color='steelblue', linestyle='--', linewidth=1.2)
    ax.axvline(gen_mean,  color='tomato',    linestyle='--', linewidth=1.2)
    ax.set_xlabel('Pixel value (normalised [0, 1])')
    ax.set_ylabel('Density')
    ax.set_title(f'Pixel distribution — {design_name}')
    ax.legend()
    fig.tight_layout()
    _save_and_show(fig, save_path, show)


# ------------------------------------------------------------------
# Panel D — Fold summary
# ------------------------------------------------------------------

def plot_fold_summary(results, save_path, show):
    """Bar chart: gen/real diversity ratio and MMD per fold."""
    designs   = [r['design']    for r in results]
    ratios    = [r['div_ratio'] for r in results]
    mmds      = [r['mmd']       for r in results]

    x = np.arange(len(designs))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Fold summary', fontsize=13)

    # Diversity ratio
    bars = ax1.bar(x, ratios, width=0.6, color='steelblue', alpha=0.8)
    ax1.axhline(1.0, color='black', linestyle='--', linewidth=1, label='ratio = 1 (perfect match)')
    ax1.axhline(0.3, color='orange', linestyle=':', linewidth=1, label='lower bound (0.3)')
    ax1.axhline(3.0, color='orange', linestyle=':', linewidth=1, label='upper bound (3.0)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(designs, rotation=20, ha='right', fontsize=9)
    ax1.set_ylabel('Gen / Real diversity ratio')
    ax1.set_title('Sample diversity ratio\n(1.0 = gen matches real spread)')
    ax1.legend(fontsize=8)
    for bar, v in zip(bars, ratios):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    # MMD
    bars2 = ax2.bar(x, mmds, width=0.6, color='tomato', alpha=0.8)
    ax2.axhline(0.1, color='green',  linestyle='--', linewidth=1, label='good  (< 0.1)')
    ax2.axhline(0.3, color='orange', linestyle='--', linewidth=1, label='moderate (< 0.3)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(designs, rotation=20, ha='right', fontsize=9)
    ax2.set_ylabel('MMD')
    ax2.set_title('MMD — generated vs real distribution\n(lower = more similar)')
    ax2.legend(fontsize=8)
    for bar, v in zip(bars2, mmds):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    _save_and_show(fig, save_path, show)


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _save_and_show(fig, path, show):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


def _mmd(real_flat, gen_flat, sigma=1.0):
    def rbf(X, Y):
        XX = (X ** 2).sum(1, keepdim=True)
        YY = (Y ** 2).sum(1, keepdim=True)
        dist = XX + YY.t() - 2 * X @ Y.t()
        return torch.exp(-dist / (2 * sigma ** 2))
    return (rbf(real_flat, real_flat).mean()
            + rbf(gen_flat,  gen_flat).mean()
            - 2 * rbf(real_flat, gen_flat).mean()).item()


def compute_mmd(real_tensor, gen_tensor):
    """Compute MMD using random projection (same method as evaluate.py)."""
    real_flat = real_tensor.view(real_tensor.size(0), -1)
    gen_flat  = gen_tensor.view(gen_tensor.size(0),  -1)
    proj_dim  = min(128, real_flat.size(1))
    torch.manual_seed(42)
    proj      = torch.randn(real_flat.size(1), proj_dim) / (real_flat.size(1) ** 0.5)
    return _mmd(real_flat @ proj, gen_flat @ proj)


# ------------------------------------------------------------------
# Per-fold evaluation
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate_fold(config, fold, model, k, temperature, save_dir, show, grid_n,
                  design_names):
    design_name = design_names[fold]
    print(f"\n{'='*60}")
    print(f"Fold {fold}  --  held-out design: {design_name}")
    print('='*60)

    out = Path(save_dir) / design_name
    out.mkdir(parents=True, exist_ok=True)

    # Collect data
    real_tensor, design_batch, hand_batch = collect_real_samples(config, fold, next(model.parameters()).device)
    gen_tensor = generate_samples(model, design_batch, hand_batch, k, temperature)

    design_np = design_batch.squeeze().cpu().numpy()   # (H, W)

    # ---- Panel A: Mean & Std ----
    print("  Plotting Panel A: Mean & Std ...")
    plot_mean_std(design_np, real_tensor, gen_tensor, design_name,
                  save_path=str(out / 'A_mean_std.png'), show=show)

    # ---- Panel B: Sample grid ----
    print("  Plotting Panel B: Sample grid ...")
    plot_sample_grid(real_tensor, gen_tensor, design_name,
                     save_path=str(out / 'B_sample_grid.png'), show=show,
                     n_per_row=grid_n)

    # ---- Panel C: Histogram ----
    print("  Plotting Panel C: Pixel histogram ...")
    plot_histogram(real_tensor, gen_tensor, design_name,
                   save_path=str(out / 'C_histogram.png'), show=show)

    # ---- Metrics for Panel D ----
    real_div  = real_tensor.view(real_tensor.size(0), -1).float().var(dim=0).mean().item()
    gen_div   = gen_tensor.var(dim=0).mean().item()
    div_ratio = gen_div / real_div if real_div > 1e-9 else float('nan')
    mmd_val   = compute_mmd(real_tensor, gen_tensor)

    print(f"  Real diversity : {real_div:.6f}")
    print(f"  Gen  diversity : {gen_div:.6f}  (ratio {div_ratio:.2f}x)")
    print(f"  MMD            : {mmd_val:.6f}")

    return {
        'fold':      fold,
        'design':    design_name,
        'real_div':  real_div,
        'gen_div':   gen_div,
        'div_ratio': div_ratio,
        'mmd':       mmd_val,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args   = parse_args()
    config = load_config(args.config)
    display_config(config)

    device       = get_device(config, cpu=args.cpu)
    model, _     = load_model(config, device)
    design_names = _resolve_design_names(config)
    k            = args.k if args.k else int(config.get('num_gen_samples', 10))
    folds        = [args.fold] if args.fold is not None else list(range(len(design_names)))

    print(f"\nGenerating {k} samples per design  |  temperature={args.temperature}")
    print(f"Output directory: {args.save}")

    all_results = []
    for fold in folds:
        result = evaluate_fold(
            config, fold, model, k, args.temperature,
            save_dir=args.save, show=args.show,
            grid_n=args.grid_n, design_names=design_names,
        )
        all_results.append(result)

    # ---- Panel D: Fold summary (only if multiple folds) ----
    if len(all_results) > 1:
        print("\nPlotting Panel D: Fold summary ...")
        plot_fold_summary(
            all_results,
            save_path=str(Path(args.save) / 'D_fold_summary.png'),
            show=args.show,
        )

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Design':<15} {'Real div':>10} {'Gen div':>10} {'Ratio':>7} {'MMD':>8}")
    print('-' * 60)
    for r in all_results:
        print(f"{r['design']:<15} {r['real_div']:>10.6f} {r['gen_div']:>10.6f} "
              f"{r['div_ratio']:>7.2f}x {r['mmd']:>8.4f}")

    if len(all_results) > 1:
        ratios = [r['div_ratio'] for r in all_results if not np.isnan(r['div_ratio'])]
        mmds   = [r['mmd']       for r in all_results]
        print('-' * 60)
        print(f"{'mean':<15} {'':>10} {'':>10} "
              f"{np.mean(ratios):>7.2f}x {np.mean(mmds):>8.4f}")


if __name__ == '__main__':
    main()
