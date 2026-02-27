"""Step-by-step visualization of the PCB elevation preprocessing pipeline.

Usage:
    python -m preprocess.visualize_steps --input-file path/to/sample.txt
    python -m preprocess.visualize_steps --input-file path/to/sample.txt --output steps.png
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from .io_utils import read_elevation
from .preprocessing import (
    detect_and_remove_outliers,
    downsample_median,
    interpolate_surface,
    smooth_gaussian,
)

ELEVATION_CMAP = "jet"  # red -> orange -> yellow -> green -> blue -> deep blue


def visualize_pipeline(
    filepath: str,
    downsample_factor: int = 4,
    outlier_grid_size: int = 8,
    outlier_z_threshold: float = 3.0,
    interp_poly_degree: int = 3,
    interp_ridge_alpha: float = 0.1,
    gaussian_sigma: float = 2.0,
    gaussian_iterations: int = 3,
    null_value: float = 9999.0,
    output_path: str = None,
) -> None:
    """Run the preprocessing pipeline step-by-step and display intermediate results."""

    # --- Step 1: Load raw data ---
    raw_data = read_elevation(filepath, null_value=null_value)

    # --- Step 2: Downsample ---
    downsampled = downsample_median(raw_data, factor=downsample_factor)

    # --- Step 3: Outlier detection ---
    outlier_result, n_outliers = detect_and_remove_outliers(
        downsampled, grid_size=outlier_grid_size, z_threshold=outlier_z_threshold,
    )
    # Find outlier positions: NaN in result but not NaN in downsampled
    outlier_mask = np.isnan(outlier_result) & ~np.isnan(downsampled)
    outlier_rows, outlier_cols = np.where(outlier_mask)

    # --- Step 4: Interpolation ---
    interpolated, n_interpolated = interpolate_surface(
        outlier_result, poly_degree=interp_poly_degree, ridge_alpha=interp_ridge_alpha,
    )

    # --- Step 5: Gaussian smoothing ---
    smoothed = smooth_gaussian(
        interpolated, sigma=gaussian_sigma, iterations=gaussian_iterations,
    )

    # --- Compute shared color range (across all elevation colormapped plots) ---
    all_values = [raw_data, downsampled, outlier_result, interpolated, smoothed]
    vmin = min(float(np.nanmin(a)) for a in all_values)
    vmax = max(float(np.nanmax(a)) for a in all_values)

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Preprocessing Pipeline Steps", fontsize=14, fontweight="bold")

    cmap = plt.get_cmap(ELEVATION_CMAP).copy()
    cmap.set_bad(color="white")  # NaN pixels shown as white

    # (1) Original data
    ax = axes[0, 0]
    im = ax.imshow(raw_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(f"1. Original ({raw_data.shape[0]}x{raw_data.shape[1]})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (2) Downsampled
    ax = axes[0, 1]
    im = ax.imshow(downsampled, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(f"2. Downsampled ({downsampled.shape[0]}x{downsampled.shape[1]})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (3) Outlier detection
    ax = axes[0, 2]
    im = ax.imshow(downsampled, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    if len(outlier_rows) > 0:
        ax.scatter(
            outlier_cols, outlier_rows,
            s=30, facecolors="none", edgecolors="red", linewidths=1.2,
        )
    ax.set_title(f"3. Outliers Detected ({n_outliers})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (4) Interpolated
    ax = axes[1, 0]
    im = ax.imshow(interpolated, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(f"4. Interpolated ({n_interpolated} filled)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (5) Smoothed
    ax = axes[1, 1]
    im = ax.imshow(smoothed, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(f"5. Smoothed (σ={gaussian_sigma}, iter={gaussian_iterations})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (6) Grayscale
    ax = axes[1, 2]
    im = ax.imshow(smoothed, cmap="gray", aspect="auto")
    ax.set_title("6. Grayscale")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved to: {output_path}")

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize each step of the PCB elevation preprocessing pipeline."
    )
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to a single .txt elevation file.")
    parser.add_argument("--output", type=str, default=None,
                        help="Save the figure to this path (e.g. steps.png).")
    parser.add_argument("--downsample-factor", type=int, default=4)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=0.1)
    parser.add_argument("--gaussian-sigma", type=float, default=2.0)
    parser.add_argument("--smooth-iterations", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        visualize_pipeline(
            filepath=args.input_file,
            downsample_factor=args.downsample_factor,
            outlier_grid_size=args.grid_size,
            outlier_z_threshold=args.z_threshold,
            interp_poly_degree=args.poly_degree,
            interp_ridge_alpha=args.ridge_alpha,
            gaussian_sigma=args.gaussian_sigma,
            gaussian_iterations=args.smooth_iterations,
            output_path=args.output,
        )
    except Exception as e:
        print(f"\n  [ERROR] Visualization failed: {e}", file=sys.stderr)
        sys.exit(1)
