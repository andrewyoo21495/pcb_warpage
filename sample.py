#!/usr/bin/env python3
"""Generate K elevation samples for a given design image and save each as an individual PNG.

Supports both CVAE and DDPM (auto-detected from checkpoint).
Supports single-design mode (--design) or batch mode (--design-dir).

Usage:
  # Single design
  python sample.py --design data/design/design_A.png --save outputs/samples_A/
  python sample.py --design data/design/design_C.png --k 16 --save outputs/samples_C/

  # Batch: process all designs in a folder (creates subfolders per design)
  python sample.py --design-dir data/design/ --k 16 --save outputs/samples/
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from utils.load_config import load_config
from utils.handcrafted_features import extract_handcrafted_features
from models import build_model


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Sample elevation images from trained model')
    parser.add_argument('--config',     type=str, default='config.txt')
    parser.add_argument('--design',     type=str, default=None,
                        help='Path to a single design image (PNG, grayscale)')
    parser.add_argument('--design-dir', type=str, default=None,
                        help='Path to a folder containing multiple design PNGs; '
                             'generates k samples for each design into separate subfolders')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override modelpath from config')
    parser.add_argument('--k',          type=int,   default=None,
                        help='Override num_gen_samples from config')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature: >1 increases diversity, <1 reduces it (default: 1.0)')
    parser.add_argument('--save',       type=str,   default=None,
                        help='Directory to save individual elevation PNG files '
                             '(default: sample_save_dir from config, or outputs/samples)')
    parser.add_argument('--denormalize', action='store_true',
                        help='Also save inverse-scaled physical values as .txt files '
                             '(reads elevation_min / elevation_max from config.txt)')
    parser.add_argument('--colormap',   type=str, default=None,
                        help='Apply a matplotlib colormap (e.g. "jet") to saved PNGs '
                             '(default: grayscale)')
    parser.add_argument('--save_txt',   action='store_true',
                        help='Save generated samples as tab-delimited .txt files '
                             'in a txt_data/ subfolder (reverse of preprocessing imaging step). '
                             'Auto-reads scaling_metadata.json from the elevation data directory; '
                             'override with --metadata or falls back to config elevation_min/max')
    parser.add_argument('--metadata',   type=str, default=None,
                        help='Path to scaling_metadata.json from preprocessing '
                             '(default: auto-detect in elevation data directory)')
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
# Image loading
# ------------------------------------------------------------------

def load_design(path: str, image_size: int) -> tuple[torch.Tensor, np.ndarray, Image.Image]:
    """Load and preprocess a design image.

    Returns:
        design_tensor : (1, 1, H, W)  float32 in [0, 1] for model input
        design_np     : (H, W)        float32 in [0, 1] for display
        design_orig   : PIL Image at original resolution (for feature extraction)
    """
    img_orig = Image.open(path).convert('L')          # original resolution
    img      = img_orig.resize((image_size, image_size), Image.LANCZOS)
    tensor   = TF.to_tensor(img)                      # (1, H, W)
    return tensor.unsqueeze(0), np.array(img, dtype=np.float32) / 255.0, img_orig


# ------------------------------------------------------------------
# Model loading helper
# ------------------------------------------------------------------

def load_model_from_checkpoint(checkpoint: dict, config: dict, device: torch.device):
    """Load a model from checkpoint, handling both CVAE and DDPM.

    For DDPM checkpoints, EMA weights are loaded for inference.
    """
    model_type = checkpoint.get('model_type', 'cvae')
    config['model_type'] = model_type

    model = build_model(config).to(device)

    if model_type == 'ddpm' and 'ema_state_dict' in checkpoint:
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
# Visualisation
# ------------------------------------------------------------------

def save_individual_samples(samples: torch.Tensor, save_dir: str,
                            colormap: str = None) -> None:
    """Save each generated elevation sample as an individual PNG.

    Args:
        samples  : (K, 1, H, W) float32 in [0, 1]
        save_dir : directory where files will be written;
                   created automatically if it does not exist
        colormap : optional matplotlib colormap name (e.g. "jet");
                   None saves as grayscale
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if colormap is not None:
        cmap = plt.get_cmap(colormap)
        samples_float = samples.squeeze(1).cpu().numpy().clip(0, 1)  # (K, H, W)
        for i, arr in enumerate(samples_float):
            rgba = cmap(arr)                                          # (H, W, 4) float [0,1]
            rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)           # drop alpha
            img_path = out_dir / f"elevation_{i + 1:04d}.png"
            Image.fromarray(rgb, mode='RGB').save(img_path)
            print(f"  Saved {img_path}")
    else:
        samples_np = (samples.squeeze(1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        for i, arr in enumerate(samples_np):
            img_path = out_dir / f"elevation_{i + 1:04d}.png"
            Image.fromarray(arr, mode='L').save(img_path)
            print(f"  Saved {img_path}")

    print(f"\nSaved {samples.size(0)} elevation images to '{out_dir}/'.")


def print_sample_stats(samples: torch.Tensor):
    """Print basic statistics of the generated samples."""
    s = samples.squeeze(1).cpu()  # (K, H, W)
    print("\n--- Generated Sample Statistics ---")
    print(f"  N samples      : {s.size(0)}")
    print(f"  Mean intensity : {s.mean():.4f}")
    print(f"  Std intensity  : {s.std():.4f}")
    print(f"  Min / Max      : {s.min():.4f} / {s.max():.4f}")
    # Per-pixel variance as diversity measure
    per_pixel_var = s.var(dim=0)
    print(f"  Sample diversity (mean var): {per_pixel_var.mean():.6f}")
    print("-----------------------------------\n")


def save_denormalized_txt(
    samples: torch.Tensor,
    save_dir: str,
    elev_min: float,
    elev_max: float,
) -> None:
    """Apply the inverse min-max transform and save each sample as a .txt file.

    Forward transform (applied when preparing the dataset):
        pixel_float = (physical_value - elev_min) / (elev_max - elev_min)

    Inverse (applied here):
        physical_value = pixel_float * (elev_max - elev_min) + elev_min

    Args:
        samples  : (K, 1, H, W) float32 in [0, 1]
        save_dir : directory to write .txt files (created if absent)
        elev_min : minimum physical value used in original min-max scaling
        elev_max : maximum physical value used in original min-max scaling
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_np = samples.squeeze(1).cpu().numpy()           # (K, H, W) in [0, 1]
    physical   = samples_np * (elev_max - elev_min) + elev_min   # inverse transform

    for i, arr in enumerate(physical):
        txt_path = out_dir / f"elevation_{i + 1:04d}.txt"
        np.savetxt(str(txt_path), arr, fmt='%.6f')
        print(f"  Saved {txt_path}")

    print(f"\nSaved {len(physical)} denormalized .txt files to '{out_dir}/'.")


def save_txt_files(
    samples: torch.Tensor,
    save_dir: str,
    elev_min: float,
    elev_max: float,
) -> None:
    """Reverse the preprocessing imaging step: convert generated samples back to
    tab-delimited .txt files with physical elevation values.

    Forward transform (imaging.py):
        pixel_float = (physical_value - global_min) / (global_max - global_min)

    Inverse (applied here):
        physical_value = pixel_float * (global_max - global_min) + global_min

    Files are saved to {save_dir}/txt_data/.

    Args:
        samples  : (K, 1, H, W) float32 in [0, 1]
        save_dir : parent output directory
        elev_min : global minimum elevation used in original min-max scaling
        elev_max : global maximum elevation used in original min-max scaling
    """
    out_dir = Path(save_dir) / "txt_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_np = samples.squeeze(1).cpu().numpy()                     # (K, H, W) in [0, 1]
    physical   = samples_np * (elev_max - elev_min) + elev_min        # inverse transform

    for i, arr in enumerate(physical):
        txt_path = out_dir / f"elevation_{i + 1:04d}.txt"
        np.savetxt(str(txt_path), arr, fmt='%.6f', delimiter='\t')
        print(f"  Saved {txt_path}")

    print(f"\nSaved {len(physical)} .txt files to '{out_dir}/'.")


# ------------------------------------------------------------------
# Scaling metadata helpers
# ------------------------------------------------------------------

METADATA_FILENAME = "scaling_metadata.json"

def _find_metadata(config: dict, metadata_override: str = None) -> str | None:
    """Locate scaling_metadata.json, checking in order:
    1. Explicit --metadata path
    2. elevation_data_dir from config (e.g. data/elevation/)
    3. Common default paths
    """
    if metadata_override:
        p = Path(metadata_override)
        if p.is_file():
            return str(p)
        print(f"Warning: --metadata path not found: {metadata_override}")
        return None

    # Try elevation data directory from config
    for key in ('elevation_base_dir', 'elevation_data_dir'):
        elev_dir = config.get(key, None)
        if elev_dir:
            p = Path(elev_dir) / METADATA_FILENAME
            if p.is_file():
                return str(p)

    # Try common default locations
    for candidate in ['data/elevation', 'data']:
        p = Path(candidate) / METADATA_FILENAME
        if p.is_file():
            return str(p)

    return None


def load_scaling_metadata(config: dict, metadata_override: str = None) -> tuple[float, float] | None:
    """Load scaling range from scaling_metadata.json.

    Returns (scale_min, scale_max) or None if not found.
    Supports both new keys (scale_min/scale_max) and legacy keys (global_min/global_max).
    """
    path = _find_metadata(config, metadata_override)
    if path is None:
        return None

    with open(path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    # Prefer fixed scaling keys; fall back to legacy global_min/global_max
    scale_min = float(meta.get('scale_min', meta.get('global_min')))
    scale_max = float(meta.get('scale_max', meta.get('global_max')))
    print(f"Loaded scaling metadata from {path}  "
          f"(scale_min={scale_min:.4f}, scale_max={scale_max:.4f})")
    return scale_min, scale_max


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def plot_generated_histograms(all_samples: dict, base_save_dir,
                              elev_min: float = 0.0,
                              elev_max: float = 1.0) -> None:
    """Plot elevation range distribution histograms for generated samples.

    For each design, applies inverse min-max scaling to recover physical
    elevation values, computes per-sample elevation range (max - min),
    and saves a histogram to outputs/distribution/generated/.

    Args:
        all_samples : dict mapping design_name -> (K, 1, H, W) tensor
        base_save_dir : Path to the base output directory
        elev_min : global minimum elevation used in preprocessing scaling
        elev_max : global maximum elevation used in preprocessing scaling
    """
    hist_dir = Path("outputs/distribution/generated")
    hist_dir.mkdir(parents=True, exist_ok=True)

    scale = elev_max - elev_min

    for design_name, samples in all_samples.items():
        samples_np = samples.squeeze(1).cpu().numpy()  # (K, H, W)
        # Inverse min-max scaling: physical = normalized * (max - min) + min
        samples_physical = samples_np * scale + elev_min
        ranges = [arr.max() - arr.min() for arr in samples_physical]
        ranges = np.array(ranges)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(ranges, bins=max(10, len(ranges) // 2), color='steelblue',
                edgecolor='white', alpha=0.85)
        ax.set_title(f"Elevation Range Distribution — {design_name}")
        ax.set_xlabel("Elevation Range (max − min)")
        ax.set_ylabel("Count")
        ax.axvline(ranges.mean(), color='red', linestyle='--', linewidth=1.2,
                   label=f"Mean = {ranges.mean():.4f}\nMin = {ranges.min():.4f}  |  Max = {ranges.max():.4f}")
        ax.legend()
        fig.tight_layout()

        out_path = hist_dir / f"dist_{design_name}.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"  Saved histogram: {out_path}")


def _resolve_scaling(args, config):
    """Resolve elevation scaling parameters from metadata or config."""
    scaling = load_scaling_metadata(config, args.metadata)
    if scaling is not None:
        return scaling
    elev_min = float(config.get('elevation_min', 0.0))
    elev_max = float(config.get('elevation_max', 1.0))
    if elev_min == 0.0 and elev_max == 1.0:
        print("Warning: No scaling_metadata.json found and elevation_min/elevation_max "
              "are at default [0.0, 1.0]. Set them in config.txt or re-run "
              "preprocessing to generate scaling_metadata.json.")
    else:
        print(f"Using elevation_min/max from config.txt")
    return elev_min, elev_max


def generate_for_design(design_path: str, model, model_type: str, config: dict,
                        device: torch.device, args, save_dir: str) -> torch.Tensor:
    """Generate and save k samples for a single design image.

    Returns the generated samples tensor (K, 1, H, W).
    """
    image_size = int(config.get('image_size', 256))
    k = args.k if args.k else int(config.get('num_gen_samples', 10))

    design_tensor, design_np, design_orig = load_design(design_path, image_size)
    design_tensor = design_tensor.to(device)

    hand_features = extract_handcrafted_features(design_orig)
    hand_features = hand_features.unsqueeze(0).to(device)

    use_amp = device.type == 'cuda'
    print(f"Generating {k} elevation samples for {design_path}  "
          f"(temperature={args.temperature}, model={model_type.upper()}) ...")
    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        samples = model.sample(design_tensor, hand_features,
                               num_samples=k, temperature=args.temperature)
    samples = samples.float()

    print_sample_stats(samples)
    save_individual_samples(samples, save_dir=save_dir, colormap=args.colormap)

    if args.save_txt:
        elev_min, elev_max = _resolve_scaling(args, config)
        print(f"\nConverting to .txt files with physical range [{elev_min}, {elev_max}] ...")
        save_txt_files(samples, save_dir=save_dir,
                       elev_min=elev_min, elev_max=elev_max)

    if args.denormalize:
        elev_min = float(config.get('elevation_min', 0.0))
        elev_max = float(config.get('elevation_max', 1.0))
        if elev_min == 0.0 and elev_max == 1.0:
            print("Warning: elevation_min/elevation_max are both at default [0.0, 1.0]. "
                  "Set them in config.txt to obtain physically meaningful values.")
        print(f"\nDenormalizing to physical range [{elev_min}, {elev_max}] ...")
        save_denormalized_txt(samples, save_dir=save_dir,
                              elev_min=elev_min, elev_max=elev_max)

    return samples


def main():
    args   = parse_args()
    config = load_config(args.config)

    if not args.design and not args.design_dir:
        raise ValueError("Specify either --design (single image) or --design-dir (folder of images).")

    device     = get_device(config)

    # Checkpoint
    model_path = args.checkpoint or config.get('modelpath', './outputs/cvae_pcb.pth')
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}\n"
                                "Run train.py first.")

    # Load model (auto-detects CVAE or DDPM from checkpoint)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model, model_type = load_model_from_checkpoint(checkpoint, config, device)
    print(f"Loaded {model_type.upper()} model from {model_path}  "
          f"(epoch {checkpoint.get('epoch', '?')})")

    save_dir = args.save or config.get('sample_save_dir', 'outputs/samples')

    if args.design_dir:
        # --- Batch mode: iterate all design PNGs in the folder ---
        design_dir = Path(args.design_dir)
        if not design_dir.is_dir():
            raise FileNotFoundError(f"Design directory not found: {design_dir}")

        design_files = sorted(design_dir.glob("*.png"))
        if not design_files:
            raise FileNotFoundError(f"No .png files found in {design_dir}")

        print(f"\nBatch mode: found {len(design_files)} design images in '{design_dir}'")
        print("=" * 60)

        base_save_dir = Path(save_dir)
        all_samples = {}

        for design_path in design_files:
            design_name = design_path.stem  # e.g. "design_A"
            save_dir = str(base_save_dir / design_name)

            print(f"\n{'─' * 60}")
            print(f"  Design: {design_path.name}  →  {save_dir}/")
            print(f"{'─' * 60}")

            samples = generate_for_design(
                str(design_path), model, model_type, config, device, args, save_dir
            )
            all_samples[design_name] = samples

        # Generate elevation range histograms for generated data (in physical scale)
        elev_min, elev_max = _resolve_scaling(args, config)
        print(f"\n{'=' * 60}")
        print("  Generating elevation range histograms for generated samples...")
        print(f"  (physical scale: [{elev_min:.4f}, {elev_max:.4f}])")
        plot_generated_histograms(all_samples, base_save_dir,
                                  elev_min=elev_min, elev_max=elev_max)

        print(f"\n{'=' * 60}")
        print(f"  Batch complete: processed {len(design_files)} designs.")
        print(f"{'=' * 60}\n")

    else:
        # --- Single design mode (original behaviour) ---
        generate_for_design(args.design, model, model_type, config, device, args, save_dir)


if __name__ == '__main__':
    main()
