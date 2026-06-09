#!/usr/bin/env python3
"""DF²M Evaluation Script — Leave-one-out evaluation.

Evaluates each module independently and the full pipeline:
    1. Module A: Mean prediction MSE on held-out design
    2. Module B-1: Residual CAE reconstruction quality
    3. Full pipeline: Generated sample quality (MSE, Diversity, MMD)

Usage:
    python dfm_approach/evaluate.py --config dfm_approach/config_dfm.txt --fold 0 --k 50
    python dfm_approach/evaluate.py --config dfm_approach/config_dfm.txt --all-folds --k 50
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.load_config import load_config, display_config
from utils.handcrafted_features import extract_handcrafted_features
from utils.ema import EMA

from models.fno_mean_predictor import FNOMeanPredictor
from models.condition_encoder import ConditionEncoder
from models.residual_cae import ResidualCAE
from models.ot_cfm import OTCFM


def _select_features(features, selected):
    if selected is not None and len(selected) < features.shape[-1]:
        return features[..., selected]
    return features


# ------------------------------------------------------------------
# Load all models
# ------------------------------------------------------------------

def load_all_models(config, device):
    """Load trained FNO, ConditionEncoder, CAE, and CFM from checkpoints."""
    selected = config.get('selected_features', None)
    if isinstance(selected, str):
        selected = [int(x) for x in selected.split(',')]
    config['selected_features'] = selected

    # FNO
    fno = FNOMeanPredictor(config).to(device)
    fno_path = config.get('fno_modelpath', './outputs/dfm_fno.pth')
    ckpt = torch.load(fno_path, map_location=device, weights_only=False)
    fno.load_state_dict(ckpt['model_state'])
    fno.eval()

    # ConditionEncoder + CAE
    cond_enc = ConditionEncoder(config).to(device)
    cae = ResidualCAE(config).to(device)
    cae_path = config.get('cae_modelpath', './outputs/dfm_cae.pth')
    ckpt = torch.load(cae_path, map_location=device, weights_only=False)
    cond_enc.load_state_dict(ckpt['cond_enc_state'])
    cae.load_state_dict(ckpt['cae_state'])
    cond_enc.eval()
    cae.eval()

    # CFM
    cfm = OTCFM(config).to(device)
    cfm_path = config.get('cfm_modelpath', './outputs/dfm_cfm.pth')
    ckpt = torch.load(cfm_path, map_location=device, weights_only=False)
    cfm.velocity_net.load_state_dict(ckpt['cfm_state'])
    # Apply EMA weights
    if 'ema_state' in ckpt:
        ema = EMA(cfm.velocity_net, decay=0.9999)
        ema.load_shadow(ckpt['ema_state'])
        ema.apply_shadow()
    cfm.eval()

    return fno, cond_enc, cae, cfm


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_mmd(real: torch.Tensor, generated: torch.Tensor, sigma: float = 1.0) -> float:
    """Maximum Mean Discrepancy with RBF kernel.

    Args:
        real:      (N, D) flattened real images
        generated: (M, D) flattened generated images
    """
    # Project to lower dimension for efficiency
    D = real.shape[1]
    if D > 256:
        proj = torch.randn(D, 256, device=real.device) / (256 ** 0.5)
        real = real @ proj
        generated = generated @ proj

    def rbf_kernel(x, y):
        xx = (x * x).sum(dim=1, keepdim=True)
        yy = (y * y).sum(dim=1, keepdim=True)
        dist = xx + yy.T - 2.0 * x @ y.T
        return torch.exp(-dist / (2.0 * sigma ** 2))

    kxx = rbf_kernel(real, real).mean()
    kyy = rbf_kernel(generated, generated).mean()
    kxy = rbf_kernel(real, generated).mean()
    return float(kxx + kyy - 2 * kxy)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate_fold(
    config: dict,
    fold: int,
    k: int,
    device: torch.device,
):
    """Evaluate DF²M on a single fold.

    Returns metrics dict.
    """
    config['val_fold'] = fold
    fno, cond_enc, cae, cfm = load_all_models(config, device)

    # Resolve paths
    design_names_raw = config.get('design_names', None)
    if design_names_raw is None:
        design_names = [f'design_{chr(65+i)}' for i in range(10)]
    elif isinstance(design_names_raw, list):
        design_names = [str(n).strip() for n in design_names_raw]
    else:
        design_names = [str(n).strip() for n in str(design_names_raw).split(',')]

    val_name = design_names[fold]
    dataset_dir = Path(config.get('dataset_dir', './data'))
    design_image_dir = Path(config.get('design_image_dir', str(dataset_dir / 'design')))
    elevation_base_dir = Path(config.get('elevation_base_dir', str(dataset_dir / 'elevation')))
    elevation_subdir = str(config.get('elevation_subdir', '')).strip()
    image_size = int(config.get('image_size', 128))
    size = (image_size, image_size)

    selected = config.get('selected_features', None)

    print(f"\n{'='*60}")
    print(f"Evaluating fold {fold}: held-out design = '{val_name}'")
    print(f"{'='*60}")

    # --- Load held-out design ---
    design_path = design_image_dir / f"{val_name}.png"
    design_pil = Image.open(str(design_path)).convert('L')
    hand_feat = extract_handcrafted_features(design_pil)
    hand_feat = _select_features(hand_feat, selected)

    design_tensor = TF.to_tensor(design_pil.resize(size, Image.LANCZOS)).unsqueeze(0).to(device)
    feat_tensor = hand_feat.unsqueeze(0).to(device)

    # --- Module A: Mean prediction ---
    pred_mean = fno.predict(design_tensor, feat_tensor)  # (1, 1, H, W)

    # Load real elevations for comparison
    if elevation_subdir:
        elev_dir = elevation_base_dir / val_name / elevation_subdir
    else:
        elev_dir = elevation_base_dir / val_name
    elev_paths = sorted(elev_dir.glob('*.png'))

    real_elevations = []
    for ep in elev_paths:
        e = TF.to_tensor(Image.open(str(ep)).convert('L').resize(size, Image.LANCZOS))
        real_elevations.append(e)
    real_stack = torch.stack(real_elevations).to(device)  # (N, 1, H, W)
    real_mean = real_stack.mean(dim=0, keepdim=True)       # (1, 1, H, W)

    mean_mse = F.mse_loss(pred_mean, real_mean).item()
    print(f"  Module A — Mean prediction MSE: {mean_mse:.6f}")

    # --- Module B: Condition encoding ---
    c_global, c_spatial = cond_enc(design_tensor, feat_tensor)

    # --- Full pipeline: Generate K samples ---
    generated = []
    batch_gen = min(k, 64)
    for start in range(0, k, batch_gen):
        n = min(batch_gen, k - start)
        z_latent = cfm.sample(c_global, c_spatial, num_samples=n)
        residual_samples = cae.decode(z_latent, c_global.expand(n, -1), c_spatial.expand(n, -1, -1, -1))
        warpage_samples = (pred_mean.expand(n, -1, -1, -1) + residual_samples).clamp(0, 1)
        generated.append(warpage_samples)

    gen_stack = torch.cat(generated, dim=0)  # (K, 1, H, W)

    # --- Metrics ---
    # Full sample MSE (mean of generated vs real mean)
    gen_mean = gen_stack.mean(dim=0, keepdim=True)
    full_mse = F.mse_loss(gen_mean, real_mean).item()

    # Per-sample MSE against real mean
    per_sample_mse = F.mse_loss(
        gen_stack, real_mean.expand_as(gen_stack)
    ).item()

    # Diversity: per-pixel variance
    gen_diversity = gen_stack.var(dim=0).mean().item()
    real_diversity = real_stack.var(dim=0).mean().item()
    diversity_ratio = gen_diversity / max(real_diversity, 1e-8)

    # MMD
    real_flat = real_stack.flatten(1)
    gen_flat = gen_stack.flatten(1)
    mmd = compute_mmd(real_flat, gen_flat)

    # Spectral divergence
    fft_real = torch.fft.rfft2(real_stack.float().mean(dim=0), norm='ortho').abs()
    fft_gen = torch.fft.rfft2(gen_stack.float().mean(dim=0), norm='ortho').abs()
    spectral_div = F.mse_loss(fft_gen, fft_real).item()

    metrics = {
        'fold': fold,
        'design': val_name,
        'mean_mse': mean_mse,
        'full_mse': full_mse,
        'per_sample_mse': per_sample_mse,
        'gen_diversity': gen_diversity,
        'real_diversity': real_diversity,
        'diversity_ratio': diversity_ratio,
        'mmd': mmd,
        'spectral_div': spectral_div,
        'num_real': len(real_elevations),
        'num_gen': k,
    }

    print(f"  Full pipeline — Gen mean MSE: {full_mse:.6f}")
    print(f"  Diversity ratio: {diversity_ratio:.4f} (gen={gen_diversity:.6f}, real={real_diversity:.6f})")
    print(f"  MMD: {mmd:.6f}")
    print(f"  Spectral divergence: {spectral_div:.6f}")

    # --- Save visualisation ---
    vis_dir = Path(config.get('vis_save_dir', './outputs/vis_dfm'))
    os.makedirs(str(vis_dir), exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'Fold {fold}: {val_name} | Mean MSE={mean_mse:.4f} | MMD={mmd:.4f}', fontsize=14)

        # Row 1: Real mean, Predicted mean, Residual example, Generated samples
        axes[0, 0].imshow(real_mean.squeeze().cpu(), cmap='viridis')
        axes[0, 0].set_title('Real Mean')
        axes[0, 1].imshow(pred_mean.squeeze().cpu(), cmap='viridis')
        axes[0, 1].set_title('Pred Mean (FNO)')
        diff = (pred_mean - real_mean).squeeze().cpu()
        axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-diff.abs().max(), vmax=diff.abs().max())
        axes[0, 2].set_title('Mean Error')
        for i in range(2):
            axes[0, 3 + i].imshow(gen_stack[i].squeeze().cpu(), cmap='viridis')
            axes[0, 3 + i].set_title(f'Gen Sample {i+1}')

        # Row 2: Real samples + variance maps
        for i in range(2):
            axes[1, i].imshow(real_stack[i].squeeze().cpu(), cmap='viridis')
            axes[1, i].set_title(f'Real Sample {i+1}')
        axes[1, 2].imshow(real_stack.var(dim=0).squeeze().cpu(), cmap='hot')
        axes[1, 2].set_title('Real Variance')
        axes[1, 3].imshow(gen_stack.var(dim=0).squeeze().cpu(), cmap='hot')
        axes[1, 3].set_title('Gen Variance')
        axes[1, 4].imshow(design_tensor.squeeze().cpu(), cmap='gray')
        axes[1, 4].set_title('Design')

        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(str(vis_dir / f'eval_fold{fold}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Visualisation saved: {vis_dir / f'eval_fold{fold}.png'}")
    except ImportError:
        print("  matplotlib not available, skipping visualisation")

    return metrics


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='DF²M Evaluation')
    parser.add_argument('--config', type=str, default='dfm_approach/config_dfm.txt')
    parser.add_argument('--fold', type=int, default=0, help='Fold index to evaluate')
    parser.add_argument('--all-folds', action='store_true', help='Evaluate all folds')
    parser.add_argument('--k', type=int, default=50, help='Number of samples to generate per design')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device index (overrides config)')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs (for shell script compatibility; '
                             'evaluation uses single GPU per fold)')
    parser.add_argument('--tag', type=str, default=None, help='Tag for fold-specific checkpoints')
    args = parser.parse_args()

    config = load_config(args.config)

    # Tag-based checkpoint paths (match train.py naming)
    if args.tag:
        tag = args.tag
        for key in ('fno_modelpath', 'cae_modelpath', 'cfm_modelpath'):
            path = config.get(key, '')
            if path:
                base, ext = os.path.splitext(path)
                config[key] = f"{base}_{tag}{ext}"

    selected = config.get('selected_features', None)
    if isinstance(selected, str):
        selected = [int(x) for x in selected.split(',')]
    config['selected_features'] = selected

    if args.gpu is not None:
        gpu_id = args.gpu
    else:
        gpu_ids = config.get('gpu_ids', 0)
        gpu_id = gpu_ids[0] if isinstance(gpu_ids, list) else gpu_ids
    device = torch.device(f'cuda:{gpu_id}' if gpu_id >= 0 and torch.cuda.is_available() else 'cpu')

    design_names_raw = config.get('design_names', None)
    if design_names_raw is None:
        n_designs = int(config.get('num_designs', 10))
    elif isinstance(design_names_raw, list):
        n_designs = len(design_names_raw)
    else:
        n_designs = len(str(design_names_raw).split(','))

    folds = range(n_designs) if args.all_folds else [args.fold]

    all_metrics = []
    for fold in folds:
        metrics = evaluate_fold(config, fold, args.k, device)
        all_metrics.append(metrics)

    # Summary
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY ACROSS ALL FOLDS")
        print(f"{'='*60}")
        for key in ['mean_mse', 'full_mse', 'diversity_ratio', 'mmd', 'spectral_div']:
            values = [m[key] for m in all_metrics]
            print(f"  {key}: mean={np.mean(values):.6f} ± std={np.std(values):.6f}")


if __name__ == '__main__':
    main()
