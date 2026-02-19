#!/usr/bin/env python3
"""Visualise synthetic data: plot design + sample elevation pairs per design.

Run:
  python -m data_generation.visualize_samples
  (reads data/design/*.png and data/elevation/design_*/*.png)
"""

import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR    = Path(__file__).parent.parent / 'data'
DESIGN_DIR  = DATA_DIR / 'design'
ELEV_DIR    = DATA_DIR / 'elevation'
DESIGN_NAMES = ['design_A', 'design_B', 'design_C', 'design_D']

N_ELEV_COLS = 4   # how many elevation samples to show per design row
SEED        = 0


def load_gray(path: Path) -> np.ndarray:
    """Load a grayscale PNG and return a float32 array in [0, 1]."""
    return np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0


def visualize(
    design_dir: Path  = DESIGN_DIR,
    elev_dir: Path    = ELEV_DIR,
    n_elev_cols: int  = N_ELEV_COLS,
    seed: int         = SEED,
    save_path: str    = None,
):
    """Plot grid: rows = designs, cols = design image + N elevation samples."""
    random.seed(seed)

    n_rows = len(DESIGN_NAMES)
    n_cols = 1 + n_elev_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )
    fig.suptitle("Synthetic PCB Data — Design + Elevation Samples", fontsize=13, y=1.01)

    for row, name in enumerate(DESIGN_NAMES):
        design_path = design_dir / f"{name}.png"
        elev_subdir = elev_dir / name

        # --- Design image (column 0) ---
        ax = axes[row][0]
        if design_path.exists():
            img = load_gray(design_path)
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            density = (img < 0.5).mean() * 100
            ax.set_title(f"{name}\n(density {density:.1f}%)", fontsize=8)
        else:
            ax.text(0.5, 0.5, 'not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name, fontsize=8)
        ax.axis('off')

        # --- Elevation samples (columns 1 …) ---
        if elev_subdir.exists():
            all_elevs = sorted(elev_subdir.glob('*.png'))
            chosen    = random.sample(all_elevs, min(n_elev_cols, len(all_elevs)))
        else:
            chosen = []

        for col in range(1, n_cols):
            ax = axes[row][col]
            idx = col - 1
            if idx < len(chosen):
                img = load_gray(chosen[idx])
                im  = ax.imshow(img, cmap='hot', vmin=0, vmax=1)
                mean_val = img.mean()
                ax.set_title(f"elev sample\nmean={mean_val:.2f}", fontsize=7)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9)
            ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def plot_elevation_variance(
    elev_dir: Path = ELEV_DIR,
    max_samples: int = 50,
    seed: int = SEED,
):
    """Plot per-pixel variance across elevation samples for each design."""
    random.seed(seed)
    n = len(DESIGN_NAMES)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    fig.suptitle("Elevation Sample Variance per Design", fontsize=12)

    for ax, name in zip(axes, DESIGN_NAMES):
        elev_subdir = elev_dir / name
        if not elev_subdir.exists():
            ax.set_title(f"{name}\n(no data)")
            continue

        paths  = sorted(elev_subdir.glob('*.png'))[:max_samples]
        stack  = np.stack([load_gray(p) for p in paths], axis=0)  # (N, H, W)
        var_map = stack.var(axis=0)

        im = ax.imshow(var_map, cmap='viridis')
        ax.set_title(f"{name}\nmean_var={var_map.mean():.4f}", fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Plotting design + elevation pairs...")
    visualize()
    print("Plotting elevation variance maps...")
    plot_elevation_variance()
