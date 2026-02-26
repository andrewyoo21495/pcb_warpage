"""
PCB Elevation Data Preprocessor — Standalone Single-File Version
================================================================

A self-contained pipeline that batch-preprocesses PCB warpage (elevation)
measurement data: noise removal, interpolation, smoothing, and grayscale
image generation.

Requirements:
    pip install numpy scipy scikit-learn Pillow

Usage:
    # Batch mode — process all subfolders under a root directory
    python preprocess_total.py --root-dir /path/to/data

    # Single-file mode — process one .txt file
    python preprocess_total.py --input-file /path/to/sample.txt

    # With parallel processing (e.g. 4 workers)
    python preprocess_total.py --root-dir /path/to/data --workers 4

    # With custom parameters
    python preprocess_total.py --root-dir /path/to/data --downsample-factor 4 \
        --z-threshold 3.0 --poly-degree 3 --ridge-alpha 0.1 --gaussian-sigma 1.0
"""

import argparse
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

EXCLUDE_SUFFIXES = ("ORI", "ORI@LOW", "ORI_A")


@dataclass
class PreprocessorConfig:
    """Configuration for the PCB elevation preprocessing pipeline."""

    # Input mode (mutually exclusive)
    root_dir: Optional[str] = None
    input_file: Optional[str] = None

    # File filtering
    exclude_suffixes: List[str] = field(
        default_factory=lambda: ["ORI", "ORI@LOW", "ORI_A"]
    )

    # Null value sentinel
    null_value: float = 9999.0

    # Downsampling
    downsample_factor: int = 4

    # Outlier detection
    outlier_z_threshold: float = 3.0
    outlier_grid_size: int = 8

    # Polynomial interpolation
    interp_poly_degree: int = 3
    interp_ridge_alpha: float = 0.1

    # Gaussian smoothing
    gaussian_sigma: float = 1.0

    # Parallel processing
    max_workers: int = 1

    # Output
    image_format: str = "png"
    colormap: str = "gray"


# =============================================================================
# File I/O Utilities
# =============================================================================

def should_skip_file(filepath: str, exclude_suffixes: list = EXCLUDE_SUFFIXES) -> bool:
    """Returns True if the filename (without extension) ends with any of the exclude_suffixes."""
    stem = Path(filepath).stem
    return any(stem.endswith(suffix) for suffix in exclude_suffixes)


def read_elevation(filepath: str, null_value: float = 9999.0) -> np.ndarray:
    """Load a tab-delimited elevation file and replace null sentinels with NaN."""
    data = np.loadtxt(filepath, delimiter='\t')
    data[data == null_value] = np.nan
    return data


def save_preprocessed_txt(data: np.ndarray, output_path: str) -> None:
    """Save preprocessed data as tab-delimited text with 4 decimal places."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savetxt(output_path, data, delimiter='\t', fmt='%.4f')


def discover_subfolders(root_dir: str) -> List[str]:
    """Return a sorted list of immediate subdirectory paths under root_dir."""
    subfolders = []
    for entry in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, entry)
        if os.path.isdir(full_path):
            if entry in ("interpolated", "images"):
                continue
            subfolders.append(full_path)
    return subfolders


def discover_txt_files(subfolder: str, exclude_suffixes: list) -> List[str]:
    """Return sorted list of .txt file paths in subfolder, excluding output dirs and skip-suffixes."""
    txt_files = []
    for entry in sorted(os.listdir(subfolder)):
        if not entry.lower().endswith('.txt'):
            continue
        full_path = os.path.join(subfolder, entry)
        if not os.path.isfile(full_path):
            continue
        if should_skip_file(full_path, exclude_suffixes):
            continue
        txt_files.append(full_path)
    return txt_files


def discover_preprocessed_files(root_dir: str) -> List[str]:
    """Find all *_preprocessed.txt files under root_dir/*/interpolated/."""
    results = []
    for subfolder in discover_subfolders(root_dir):
        interp_dir = os.path.join(subfolder, "interpolated")
        if not os.path.isdir(interp_dir):
            continue
        for entry in sorted(os.listdir(interp_dir)):
            if entry.endswith("_preprocessed.txt"):
                results.append(os.path.join(interp_dir, entry))
    return results


def get_output_paths(filepath: str, parent_dir: str) -> Tuple[str, str]:
    """Compute output paths for preprocessed txt and image.

    Returns:
        (txt_output_path, img_output_path)
    """
    stem = Path(filepath).stem
    txt_path = os.path.join(parent_dir, "interpolated", f"{stem}_preprocessed.txt")
    img_path = os.path.join(parent_dir, "images", f"{stem}_preprocessed.png")
    return txt_path, img_path


# =============================================================================
# Preprocessing Functions
# =============================================================================

def downsample_median(data: np.ndarray, factor: int) -> np.ndarray:
    """Downsample data by computing the median of non-NaN values in each block.

    Args:
        data: Input array of shape (H, W), may contain NaN.
        factor: Downsampling factor (e.g., 4 means 1/4 resolution).

    Returns:
        Downsampled array of shape (H // factor, W // factor).
    """
    H, W = data.shape
    new_H = H // factor
    new_W = W // factor

    # Truncate to exact multiple of factor
    trimmed = data[:new_H * factor, :new_W * factor]

    # Vectorized: reshape into blocks and compute nanmedian in one call
    reshaped = trimmed.reshape(new_H, factor, new_W, factor)
    reshaped = reshaped.transpose(0, 2, 1, 3).reshape(new_H, new_W, factor * factor)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmedian(reshaped, axis=2)

    return result


def detect_and_remove_outliers(
    data: np.ndarray,
    grid_size: int = 8,
    z_threshold: float = 3.0,
) -> tuple:
    """Detect outliers per region using z-score and replace with NaN.

    Returns:
        (result_array, total_outliers_removed)
    """
    result = data.copy()
    H, W = result.shape
    total_removed = 0

    row_edges = np.linspace(0, H, grid_size + 1, dtype=int)
    col_edges = np.linspace(0, W, grid_size + 1, dtype=int)

    for ri in range(grid_size):
        for ci in range(grid_size):
            r_start, r_end = row_edges[ri], row_edges[ri + 1]
            c_start, c_end = col_edges[ci], col_edges[ci + 1]
            region = result[r_start:r_end, c_start:c_end]

            valid_mask = ~np.isnan(region)
            valid_vals = region[valid_mask]

            if len(valid_vals) < 2:
                continue

            mean = np.mean(valid_vals)
            std = np.std(valid_vals, ddof=0)

            if std == 0:
                continue

            z_scores = np.abs((region - mean) / std)
            outlier_mask = valid_mask & (z_scores > z_threshold)
            n_outliers = np.sum(outlier_mask)

            if n_outliers > 0:
                region[outlier_mask] = np.nan
                total_removed += n_outliers

    return result, int(total_removed)


def interpolate_surface(
    data: np.ndarray,
    poly_degree: int = 3,
    ridge_alpha: float = 0.1,
) -> tuple:
    """Fill NaN values using polynomial surface regression with ridge regularization.

    Returns:
        (result_array, n_interpolated) — array with NaNs filled, and count of filled pixels.
    """
    H, W = data.shape
    total_pixels = H * W

    rows, cols = np.indices((H, W))
    rows_flat = rows.ravel().astype(np.float64)
    cols_flat = cols.ravel().astype(np.float64)
    vals_flat = data.ravel()

    valid_mask = ~np.isnan(vals_flat)
    nan_mask = np.isnan(vals_flat)

    n_valid = np.sum(valid_mask)
    n_interpolated = int(np.sum(nan_mask))
    valid_ratio = n_valid / total_pixels

    if n_valid == 0:
        logger.warning("No valid values — cannot interpolate.")
        return data, 0

    if valid_ratio < 0.05:
        logger.warning("Valid values: %.1f%% (< 5%%) — potential quality degradation.",
                        valid_ratio * 100)

    # Normalize coordinates for numerical stability
    row_norm = rows_flat / max(H - 1, 1)
    col_norm = cols_flat / max(W - 1, 1)
    coords = np.column_stack([row_norm, col_norm])

    # Polynomial features
    poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
    X_all = poly.fit_transform(coords)

    X_train = X_all[valid_mask]
    y_train = vals_flat[valid_mask]

    # Fit ridge regression
    model = Ridge(alpha=ridge_alpha)
    model.fit(X_train, y_train)

    # Predict NaN positions only
    result = data.copy()
    if np.any(nan_mask):
        X_predict = X_all[nan_mask]
        predicted = model.predict(X_predict)
        result_flat = result.ravel()
        result_flat[nan_mask] = predicted
        result = result_flat.reshape(H, W)

    return result, n_interpolated


def smooth_gaussian(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing while preserving the original min and max values.

    After smoothing, the result is linearly rescaled so that its min and max
    match the original data's min and max.
    """
    orig_min = np.min(data)
    orig_max = np.max(data)

    smoothed = gaussian_filter(data, sigma=sigma)

    smooth_min = np.min(smoothed)
    smooth_max = np.max(smoothed)

    # Rescale smoothed data to preserve original min/max
    if smooth_max - smooth_min < 1e-12:
        return smoothed

    rescaled = (smoothed - smooth_min) / (smooth_max - smooth_min)
    rescaled = rescaled * (orig_max - orig_min) + orig_min

    return rescaled


# =============================================================================
# Imaging Functions
# =============================================================================

def compute_global_minmax(root_dir: str) -> Tuple[float, float]:
    """Compute global min and max across all preprocessed files."""
    files = discover_preprocessed_files(root_dir)
    if not files:
        raise ValueError(f"No preprocessed files found under {root_dir}")

    global_min = np.inf
    global_max = -np.inf

    for fpath in files:
        data = np.loadtxt(fpath, delimiter='\t')
        file_min = np.min(data)
        file_max = np.max(data)
        global_min = min(global_min, file_min)
        global_max = max(global_max, file_max)

    return float(global_min), float(global_max)


def generate_grayscale_image(
    data: np.ndarray,
    global_min: float,
    global_max: float,
    output_path: str,
) -> None:
    """Scale data to [0, 255] using global min/max and save as grayscale image."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if global_min == global_max:
        img_array = np.full(data.shape, 128, dtype=np.uint8)
    else:
        scaled = (data - global_min) / (global_max - global_min)
        scaled = np.clip(scaled, 0.0, 1.0)
        img_array = (scaled * 255).astype(np.uint8)

    img = Image.fromarray(img_array, mode='L')
    img.save(output_path)


# =============================================================================
# Orchestration
# =============================================================================

def process_single_file(filepath: str, config: PreprocessorConfig) -> tuple:
    """Run Steps 1-5 on a single file and return the preprocessed data with stats.

    Returns:
        (data, n_outliers, n_interpolated) or (None, 0, 0) if the file should be skipped.
    """
    try:
        data = read_elevation(filepath, null_value=config.null_value)
    except Exception as e:
        logger.warning("Failed to read %s: %s", filepath, e)
        return None, 0, 0

    if np.all(np.isnan(data)):
        return None, 0, 0

    data = downsample_median(data, factor=config.downsample_factor)
    data, n_outliers = detect_and_remove_outliers(
        data,
        grid_size=config.outlier_grid_size,
        z_threshold=config.outlier_z_threshold,
    )
    data, n_interpolated = interpolate_surface(
        data,
        poly_degree=config.interp_poly_degree,
        ridge_alpha=config.interp_ridge_alpha,
    )
    data = smooth_gaussian(data, sigma=config.gaussian_sigma)

    return data, n_outliers, n_interpolated


def _process_and_save(filepath: str, output_dir: str, config: PreprocessorConfig) -> tuple:
    """Process a single file and save outputs.

    Returns:
        (filepath, n_outliers, n_interpolated) on success, or (None, 0, 0) on failure.
    """
    data, n_outliers, n_interpolated = process_single_file(filepath, config)
    if data is None:
        return None, 0, 0

    txt_path, _ = get_output_paths(filepath, output_dir)
    save_preprocessed_txt(data, txt_path)
    return filepath, n_outliers, n_interpolated


def run_single_file_mode(config: PreprocessorConfig) -> None:
    """Single-file mode: process one file and generate output."""
    filepath = config.input_file
    parent_dir = os.path.dirname(filepath)
    txt_path, img_path = get_output_paths(filepath, parent_dir)

    print(f"  Processing: {Path(filepath).name}")
    data, n_outliers, n_interpolated = process_single_file(filepath, config)
    if data is None:
        print(f"  [FAILED] {filepath}")
        return

    save_preprocessed_txt(data, txt_path)

    global_min, global_max = float(np.min(data)), float(np.max(data))
    generate_grayscale_image(data, global_min, global_max, img_path)
    print(f"  Outliers removed:     {n_outliers}")
    print(f"  Points interpolated:  {n_interpolated}")
    print(f"  Min: {global_min:.4f}  Max: {global_max:.4f}")
    print(f"  [DONE] Saved to: {txt_path}")
    print(f"         Image:    {img_path}")


def run_batch_mode(config: PreprocessorConfig) -> None:
    """Batch (directory) mode: process all subfolders, then generate images."""
    root_dir = config.root_dir
    subfolders = discover_subfolders(root_dir)

    if not subfolders:
        print("  No subfolders found. Nothing to do.")
        return

    # Collect all files to process, grouped by subfolder
    all_tasks = []  # list of (filepath, subfolder)
    for subfolder in subfolders:
        txt_files = discover_txt_files(subfolder, config.exclude_suffixes)
        for filepath in txt_files:
            all_tasks.append((filepath, subfolder))

    total_files = len(all_tasks)
    if total_files == 0:
        print("  No .txt files found to process.")
        return

    print(f"  Found {total_files} files across {len(subfolders)} subfolders.\n")

    # --- Phase 1: Per-file preprocessing (Steps 1-5) ---
    print("  Phase 1/2: Preprocessing files...")
    files_processed = 0
    files_skipped = 0
    total_outliers = 0
    total_interpolated = 0

    if config.max_workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {}
            for filepath, subfolder in all_tasks:
                fut = executor.submit(_process_and_save, filepath, subfolder, config)
                futures[fut] = filepath

            for fut in as_completed(futures):
                filepath = futures[fut]
                try:
                    result_path, n_outliers, n_interp = fut.result()
                    if result_path is not None:
                        files_processed += 1
                        total_outliers += n_outliers
                        total_interpolated += n_interp
                    else:
                        files_skipped += 1
                except Exception as e:
                    files_skipped += 1
                    logger.warning("Error processing %s: %s", filepath, e)

                done = files_processed + files_skipped
                print(f"\r    [{done}/{total_files}] processed", end="", flush=True)
    else:
        # Sequential processing
        for filepath, subfolder in all_tasks:
            result_path, n_outliers, n_interp = _process_and_save(filepath, subfolder, config)
            if result_path is not None:
                files_processed += 1
                total_outliers += n_outliers
                total_interpolated += n_interp
            else:
                files_skipped += 1

            done = files_processed + files_skipped
            print(f"\r    [{done}/{total_files}] processed", end="", flush=True)

    print()  # newline after progress
    print(f"    -> {files_processed} succeeded, {files_skipped} skipped.")
    print(f"    -> Total outliers removed:    {total_outliers}")
    print(f"    -> Total points interpolated: {total_interpolated}\n")

    if files_processed == 0:
        print("  No files were preprocessed. Skipping image generation.")
        return

    # --- Phase 2: Global scaling & image generation (Steps 6-7) ---
    print("  Phase 2/2: Generating images...")

    # Compute per-subfolder and global min/max
    global_min = np.inf
    global_max = -np.inf
    subfolder_stats = []  # list of (subfolder_name, local_min, local_max, file_count)

    for subfolder in subfolders:
        interp_dir = os.path.join(subfolder, "interpolated")
        if not os.path.isdir(interp_dir):
            continue

        local_min = np.inf
        local_max = -np.inf
        file_count = 0

        for entry in sorted(os.listdir(interp_dir)):
            if not entry.endswith("_preprocessed.txt"):
                continue
            fpath = os.path.join(interp_dir, entry)
            data = np.loadtxt(fpath, delimiter='\t')
            local_min = min(local_min, float(np.min(data)))
            local_max = max(local_max, float(np.max(data)))
            file_count += 1

        if file_count > 0:
            global_min = min(global_min, local_min)
            global_max = max(global_max, local_max)
            subfolder_name = os.path.basename(subfolder)
            subfolder_stats.append((subfolder_name, local_min, local_max, file_count))

    # Display per-subfolder min/max
    print("\n    Per-subfolder statistics:")
    for name, smin, smax, cnt in subfolder_stats:
        print(f"      {name:30s}  min={smin:10.4f}  max={smax:10.4f}  ({cnt} files)")

    print(f"\n    Global min: {global_min:.4f}")
    print(f"    Global max: {global_max:.4f}\n")

    # Generate images using global min/max
    images_generated = 0
    for subfolder in subfolders:
        interp_dir = os.path.join(subfolder, "interpolated")
        images_dir = os.path.join(subfolder, "images")

        if not os.path.isdir(interp_dir):
            continue

        for entry in sorted(os.listdir(interp_dir)):
            if not entry.endswith("_preprocessed.txt"):
                continue

            fpath = os.path.join(interp_dir, entry)
            data = np.loadtxt(fpath, delimiter='\t')

            img_name = entry.replace(".txt", ".png")
            img_path = os.path.join(images_dir, img_name)
            generate_grayscale_image(data, global_min, global_max, img_path)
            images_generated += 1

    print(f"    -> {images_generated} images generated.")


def main(config: PreprocessorConfig) -> None:
    """Main orchestration: dispatch to single-file or batch mode."""
    if config.input_file and config.root_dir:
        raise ValueError("Specify either --root-dir or --input-file, not both.")
    if not config.input_file and not config.root_dir:
        raise ValueError("Specify either --root-dir or --input-file.")

    start_time = time.time()

    print("\n" + "=" * 60)
    print("  PCB Elevation Data Preprocessor")
    print("=" * 60)

    if config.input_file:
        print(f"  Mode: Single file")
        run_single_file_mode(config)
    else:
        workers_str = f"{config.max_workers} workers" if config.max_workers > 1 else "sequential"
        print(f"  Mode: Batch ({workers_str})")
        print(f"  Root: {config.root_dir}")
        run_batch_mode(config)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Completed in {elapsed:.1f} seconds.")
    print(f"{'=' * 60}\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> PreprocessorConfig:
    """Parse CLI arguments and return a PreprocessorConfig."""
    parser = argparse.ArgumentParser(
        description="PCB Elevation Data Preprocessor"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--root-dir", type=str, default=None,
                       help="Root input directory for batch processing.")
    group.add_argument("--input-file", type=str, default=None,
                       help="Single input .txt file path.")

    parser.add_argument("--downsample-factor", type=int, default=4,
                        help="Downsampling factor (default: 4).")
    parser.add_argument("--z-threshold", type=float, default=3.0,
                        help="Z-score threshold for outlier detection (default: 3.0).")
    parser.add_argument("--grid-size", type=int, default=8,
                        help="Grid divisions for outlier detection (default: 8).")
    parser.add_argument("--poly-degree", type=int, default=3,
                        help="Polynomial degree for interpolation (default: 3).")
    parser.add_argument("--ridge-alpha", type=float, default=0.1,
                        help="Ridge regularization alpha (default: 0.1).")
    parser.add_argument("--gaussian-sigma", type=float, default=1.0,
                        help="Gaussian smoothing sigma (default: 1.0).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1 = sequential).")

    args = parser.parse_args()

    return PreprocessorConfig(
        root_dir=args.root_dir,
        input_file=args.input_file,
        downsample_factor=args.downsample_factor,
        outlier_z_threshold=args.z_threshold,
        outlier_grid_size=args.grid_size,
        interp_poly_degree=args.poly_degree,
        interp_ridge_alpha=args.ridge_alpha,
        gaussian_sigma=args.gaussian_sigma,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = parse_args()
    try:
        main(config)
    except Exception as e:
        print(f"\n  [ERROR] Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)
