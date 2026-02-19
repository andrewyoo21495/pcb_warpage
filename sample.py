#!/usr/bin/env python3
"""Generate and visualise K elevation samples for a given design image.

Usage:
  python sample.py --design data/design/design_A.png
  python sample.py --design data/design/design_C.png --k 16 --save outputs/samples_C.png
  python sample.py --config config.txt --design data/design/design_B.png --k 10
"""

import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from utils.load_config import load_config
from utils.handcrafted_features import extract_handcrafted_features
from models.cvae import CVAE


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Sample elevation images from trained CVAE')
    parser.add_argument('--config',     type=str, default='config.txt')
    parser.add_argument('--design',     type=str, required=True,
                        help='Path to design image (PNG, grayscale)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override modelpath from config')
    parser.add_argument('--k',          type=int, default=None,
                        help='Override num_gen_samples from config')
    parser.add_argument('--save',       type=str, default=None,
                        help='Save figure to this path instead of displaying')
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
# Image loading
# ------------------------------------------------------------------

def load_design(path: str, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess a design image.

    Returns:
        design_tensor : (1, 1, H, W)  float32 in [0, 1] for model input
        design_np     : (H, W)        float32 in [0, 1] for display
    """
    img    = Image.open(path).convert('L')
    img    = img.resize((image_size, image_size), Image.LANCZOS)
    tensor = TF.to_tensor(img)               # (1, H, W)
    return tensor.unsqueeze(0), np.array(img, dtype=np.float32) / 255.0


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------

def plot_samples(
    design_np: np.ndarray,
    samples: torch.Tensor,
    design_name: str,
    save_path: str = None,
):
    """Plot the design and all K generated elevation samples in a grid.

    Args:
        design_np  : (H, W) float32 in [0, 1]
        samples    : (K, 1, H, W) float32 in [0, 1]
        design_name: label for the design image
        save_path  : if provided, save figure instead of showing
    """
    k = samples.size(0)
    # +1 column for the design image
    n_cols = min(k + 1, 6)
    n_rows = ((k + 1) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.8 * n_cols, 3.0 * n_rows),
                             squeeze=False)

    # Flatten axes for easy indexing
    ax_list = [axes[r][c] for r in range(n_rows) for c in range(n_cols)]

    # First cell: design image
    ax_list[0].imshow(design_np, cmap='gray', vmin=0, vmax=1)
    ax_list[0].set_title(f"Design\n({design_name})", fontsize=9)
    ax_list[0].axis('off')

    # Remaining cells: generated samples
    samples_np = samples.squeeze(1).cpu().numpy()  # (K, H, W)
    for i in range(k):
        ax = ax_list[i + 1]
        ax.imshow(samples_np[i], cmap='hot', vmin=0, vmax=1)
        ax.set_title(f"Sample {i+1}\nμ={samples_np[i].mean():.2f}", fontsize=8)
        ax.axis('off')

    # Hide unused axes
    for ax in ax_list[k + 1:]:
        ax.axis('off')

    # Colorbar annotation
    fig.suptitle(f"CVAE Elevation Samples  (K={k})", fontsize=12, y=1.01)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args   = parse_args()
    config = load_config(args.config)

    device     = get_device(config)
    image_size = int(config.get('image_size', 256))
    k          = args.k if args.k else int(config.get('num_gen_samples', 10))

    # Checkpoint
    model_path = args.checkpoint or config.get('modelpath', './outputs/cvae_pcb.pth')
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}\n"
                                "Run train.py first.")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model      = CVAE(config).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Loaded model from {model_path}  (epoch {checkpoint.get('epoch', '?')})")

    # Load design image
    design_tensor, design_np = load_design(args.design, image_size)
    design_tensor = design_tensor.to(device)    # (1, 1, H, W)

    # Compute handcrafted features
    hand_features = extract_handcrafted_features(design_tensor.squeeze(0))  # (HAND_FEATURE_DIM,)
    hand_features = hand_features.unsqueeze(0).to(device)                   # (1, HAND_FEATURE_DIM)

    # Generate samples
    print(f"Generating {k} elevation samples for {args.design} ...")
    samples = model.sample(design_tensor, hand_features, num_samples=k)  # (K, 1, H, W)

    print_sample_stats(samples)

    design_name = Path(args.design).stem
    plot_samples(design_np, samples, design_name, save_path=args.save)


if __name__ == '__main__':
    main()
