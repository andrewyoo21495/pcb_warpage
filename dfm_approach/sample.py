#!/usr/bin/env python3
"""DF²M Inference / Sampling Script.

Generate warpage samples for one or more PCB designs using the trained DF²M pipeline.

Usage:
    # Single design
    python dfm_approach/sample.py --config dfm_approach/config_dfm.txt \
        --design /path/to/design.png --num-samples 50

    # All designs in a directory
    python dfm_approach/sample.py --config dfm_approach/config_dfm.txt \
        --design-dir /path/to/designs/ --num-samples 100

    # With denormalization to physical units (μm)
    python dfm_approach/sample.py --config dfm_approach/config_dfm.txt \
        --design /path/to/design.png --denormalize
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.load_config import load_config
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


def load_pipeline(config, device):
    """Load all trained DF²M models."""
    selected = config.get('selected_features', None)
    if isinstance(selected, str):
        selected = [int(x) for x in selected.split(',')]
    config['selected_features'] = selected

    fno = FNOMeanPredictor(config).to(device)
    ckpt = torch.load(config['fno_modelpath'], map_location=device, weights_only=False)
    fno.load_state_dict(ckpt['model_state'])
    fno.eval()

    cond_enc = ConditionEncoder(config).to(device)
    cae = ResidualCAE(config).to(device)
    ckpt = torch.load(config['cae_modelpath'], map_location=device, weights_only=False)
    cond_enc.load_state_dict(ckpt['cond_enc_state'])
    cae.load_state_dict(ckpt['cae_state'])
    cond_enc.eval()
    cae.eval()

    cfm = OTCFM(config).to(device)
    ckpt = torch.load(config['cfm_modelpath'], map_location=device, weights_only=False)
    cfm.velocity_net.load_state_dict(ckpt['cfm_state'])
    if 'ema_state' in ckpt:
        ema = EMA(cfm.velocity_net, decay=0.9999)
        ema.load_shadow(ckpt['ema_state'])
        ema.apply_shadow()
    cfm.eval()

    return fno, cond_enc, cae, cfm


@torch.no_grad()
def generate_samples(
    design_path: str,
    fno, cond_enc, cae, cfm,
    config: dict,
    device: torch.device,
    num_samples: int = 50,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate warpage samples for a single design.

    Returns:
        samples: (K, 1, H, W) generated warpage samples in [0, 1]
        pred_mean: (1, 1, H, W) predicted mean warpage
    """
    image_size = int(config.get('image_size', 128))
    size = (image_size, image_size)
    selected = config.get('selected_features', None)

    # Load and process design
    design_pil = Image.open(design_path).convert('L')
    hand_feat = extract_handcrafted_features(design_pil)
    hand_feat = _select_features(hand_feat, selected)

    design_tensor = TF.to_tensor(design_pil.resize(size, Image.LANCZOS)).unsqueeze(0).to(device)
    feat_tensor = hand_feat.unsqueeze(0).to(device)

    # Module A: Predict mean
    pred_mean = fno.predict(design_tensor, feat_tensor)  # (1, 1, H, W)

    # Module B: Encode condition
    c_global, c_spatial = cond_enc(design_tensor, feat_tensor)

    # Generate residual samples in batches
    samples_list = []
    batch_size = min(num_samples, 64)

    for start in range(0, num_samples, batch_size):
        n = min(batch_size, num_samples - start)
        z_latent = cfm.sample(c_global, c_spatial, num_samples=n, temperature=temperature)
        c_g_exp = c_global.expand(n, -1)
        c_s_exp = c_spatial.expand(n, -1, -1, -1)
        residuals = cae.decode(z_latent, c_g_exp, c_s_exp)
        warpage = (pred_mean.expand(n, -1, -1, -1) + residuals).clamp(0, 1)
        samples_list.append(warpage.cpu())

    samples = torch.cat(samples_list, dim=0)
    return samples, pred_mean.cpu()


def save_samples(
    samples: torch.Tensor,
    pred_mean: torch.Tensor,
    design_name: str,
    save_dir: str,
    denormalize: bool = False,
    elev_min: float = 0.0,
    elev_max: float = 3000.0,
):
    """Save generated samples as PNG images and optional statistics."""
    out_dir = Path(save_dir) / design_name
    os.makedirs(str(out_dir), exist_ok=True)

    # Save individual samples
    for i in range(samples.shape[0]):
        img = samples[i].squeeze()  # (H, W)
        if denormalize:
            img = img * (elev_max - elev_min) + elev_min

        # Save as PNG (normalized to [0, 255])
        if denormalize:
            img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).byte()
        else:
            img_norm = (img * 255).byte()

        Image.fromarray(img_norm.numpy(), mode='L').save(
            str(out_dir / f'sample_{i:04d}.png')
        )

    # Save predicted mean
    mean_img = pred_mean.squeeze()
    if denormalize:
        mean_img = mean_img * (elev_max - elev_min) + elev_min
    mean_norm = ((mean_img - mean_img.min()) / (mean_img.max() - mean_img.min() + 1e-8) * 255).byte()
    Image.fromarray(mean_norm.numpy(), mode='L').save(str(out_dir / 'predicted_mean.png'))

    # Save statistics
    stats = {
        'num_samples': samples.shape[0],
        'pixel_mean': samples.mean().item(),
        'pixel_std': samples.std().item(),
        'diversity': samples.var(dim=0).mean().item(),
        'min': samples.min().item(),
        'max': samples.max().item(),
    }
    if denormalize:
        denorm_samples = samples * (elev_max - elev_min) + elev_min
        stats['physical_mean_um'] = denorm_samples.mean().item()
        stats['physical_std_um'] = denorm_samples.std().item()

    with open(str(out_dir / 'stats.txt'), 'w') as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

    print(f"  Saved {samples.shape[0]} samples to {out_dir}")
    return stats


def main():
    parser = argparse.ArgumentParser(description='DF²M Sampling / Inference')
    parser.add_argument('--config', type=str, default='dfm_approach/config_dfm.txt')
    parser.add_argument('--design', type=str, default=None, help='Path to single design PNG')
    parser.add_argument('--design-dir', type=str, default=None, help='Directory of design PNGs')
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--denormalize', action='store_true')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device index (overrides config)')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs (for shell script compatibility; '
                             'sampling uses single GPU per design)')
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

    num_samples = args.num_samples or int(config.get('num_gen_samples', 300))

    if args.gpu is not None:
        gpu_id = args.gpu
    else:
        gpu_ids = config.get('gpu_ids', 0)
        gpu_id = gpu_ids[0] if isinstance(gpu_ids, list) else gpu_ids
    device = torch.device(f'cuda:{gpu_id}' if gpu_id >= 0 and torch.cuda.is_available() else 'cpu')

    fno, cond_enc, cae, cfm = load_pipeline(config, device)

    save_dir = config.get('sample_save_dir', './outputs/samples_dfm')
    elev_min = float(config.get('elevation_min', 0.0))
    elev_max = float(config.get('elevation_max', 3000.0))

    # Collect design paths
    design_paths = []
    if args.design:
        design_paths.append(args.design)
    elif args.design_dir:
        design_dir = Path(args.design_dir)
        design_paths = sorted(str(p) for p in design_dir.glob('*.png'))
    else:
        # Use all designs from config
        design_image_dir = Path(config.get('design_image_dir', './data/design'))
        design_names_raw = config.get('design_names', None)
        if design_names_raw is None:
            design_names = [f'design_{chr(65+i)}' for i in range(10)]
        elif isinstance(design_names_raw, list):
            design_names = [str(n).strip() for n in design_names_raw]
        else:
            design_names = [str(n).strip() for n in str(design_names_raw).split(',')]
        design_paths = [str(design_image_dir / f"{n}.png") for n in design_names]

    print(f"Generating {num_samples} samples for {len(design_paths)} design(s)")
    print(f"Device: {device}")

    for dp in design_paths:
        if not os.path.exists(dp):
            print(f"  [SKIP] Not found: {dp}")
            continue

        design_name = Path(dp).stem
        print(f"\nProcessing: {design_name}")

        samples, pred_mean = generate_samples(
            dp, fno, cond_enc, cae, cfm, config, device,
            num_samples=num_samples, temperature=args.temperature,
        )

        save_samples(
            samples, pred_mean, design_name, save_dir,
            denormalize=args.denormalize, elev_min=elev_min, elev_max=elev_max,
        )

    print("\nSampling complete.")


if __name__ == '__main__':
    main()
