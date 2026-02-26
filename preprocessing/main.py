"""CLI entry point and orchestration for PCB Elevation Preprocessor."""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import PreprocessorConfig
from .imaging import generate_grayscale_image
from .io_utils import (
    discover_subfolders,
    discover_txt_files,
    get_output_paths,
    read_elevation,
    save_preprocessed_txt,
)
from .preprocessing import (
    detect_and_remove_outliers,
    downsample_median,
    interpolate_surface,
    smooth_gaussian,
)

logger = logging.getLogger(__name__)


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

    # Check if entire data is NaN
    if np.all(np.isnan(data)):
        logger.warning("Entire data is NaN in %s — skipping.", filepath)
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
    data = smooth_gaussian(
        data, sigma=config.gaussian_sigma, iterations=config.gaussian_iterations,
    )

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

    # Save scaling metadata alongside output
    metadata_path = os.path.join(parent_dir, "scaling_metadata.json")
    metadata = {
        "global_min": global_min,
        "global_max": global_max,
        "num_subfolders": 1,
        "num_images": 1,
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Outliers removed:     {n_outliers}")
    print(f"  Points interpolated:  {n_interpolated}")
    print(f"  Min: {global_min:.4f}  Max: {global_max:.4f}")
    print(f"  [DONE] Saved to: {txt_path}")
    print(f"         Image:    {img_path}")
    print(f"         Metadata: {metadata_path}")


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

    # Compute per-subfolder and global min/max, and per-file elevation ranges
    global_min = np.inf
    global_max = -np.inf
    subfolder_stats = []  # list of (subfolder_name, local_min, local_max, file_count, ranges)

    for subfolder in subfolders:
        interp_dir = os.path.join(subfolder, "interpolated")
        if not os.path.isdir(interp_dir):
            continue

        local_min = np.inf
        local_max = -np.inf
        file_count = 0
        local_ranges = []  # per-file (max - min) values

        for entry in sorted(os.listdir(interp_dir)):
            if not entry.endswith("_preprocessed.txt"):
                continue
            fpath = os.path.join(interp_dir, entry)
            data = np.loadtxt(fpath, delimiter='\t')
            file_min = float(np.min(data))
            file_max = float(np.max(data))
            local_min = min(local_min, file_min)
            local_max = max(local_max, file_max)
            local_ranges.append(file_max - file_min)
            file_count += 1

        if file_count > 0:
            global_min = min(global_min, local_min)
            global_max = max(global_max, local_max)
            subfolder_name = os.path.basename(subfolder)
            subfolder_stats.append((subfolder_name, local_min, local_max, file_count, local_ranges))

    # Display per-subfolder min/max
    print("\n    Per-subfolder statistics:")
    for name, smin, smax, cnt, _ in subfolder_stats:
        print(f"      {name:30s}  min={smin:10.4f}  max={smax:10.4f}  ({cnt} files)")

    print(f"\n    Global min: {global_min:.4f}")
    print(f"    Global max: {global_max:.4f}")

    # --- Elevation range distribution analysis ---
    print("\n    Elevation range distributions (per-file max - min):")
    print("    " + "-" * 76)
    print(f"      {'Subfolder':30s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'P25':>8s}"
          f"  {'P50':>8s}  {'P75':>8s}  {'Max':>8s}")
    print("    " + "-" * 76)

    all_ranges = []
    for name, _, _, cnt, ranges in subfolder_stats:
        r = np.array(ranges)
        all_ranges.extend(ranges)
        p25, p50, p75 = np.percentile(r, [25, 50, 75])
        print(f"      {name:30s}  {r.mean():8.4f}  {r.std():8.4f}  {r.min():8.4f}"
              f"  {p25:8.4f}  {p50:8.4f}  {p75:8.4f}  {r.max():8.4f}")

    if all_ranges:
        all_r = np.array(all_ranges)
        p25, p50, p75 = np.percentile(all_r, [25, 50, 75])
        print("    " + "-" * 76)
        print(f"      {'GLOBAL':30s}  {all_r.mean():8.4f}  {all_r.std():8.4f}  {all_r.min():8.4f}"
              f"  {p25:8.4f}  {p50:8.4f}  {p75:8.4f}  {all_r.max():8.4f}")
    print()

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

    # Save scaling metadata for downstream use (e.g. sample.py --save_txt)
    metadata_path = os.path.join(root_dir, "scaling_metadata.json")
    # Build range distribution summary for metadata
    global_range_stats = {}
    if all_ranges:
        all_r = np.array(all_ranges)
        g_p25, g_p50, g_p75 = np.percentile(all_r, [25, 50, 75]).tolist()
        global_range_stats = {
            "mean": float(all_r.mean()),
            "std": float(all_r.std()),
            "min": float(all_r.min()),
            "p25": g_p25,
            "median": g_p50,
            "p75": g_p75,
            "max": float(all_r.max()),
        }

    metadata = {
        "global_min": global_min,
        "global_max": global_max,
        "num_subfolders": len(subfolder_stats),
        "num_images": images_generated,
        "subfolders": [
            {
                "name": name,
                "min": smin,
                "max": smax,
                "num_files": cnt,
                "range_distribution": {
                    "mean": float(np.mean(ranges)),
                    "std": float(np.std(ranges)),
                    "min": float(np.min(ranges)),
                    "p25": float(np.percentile(ranges, 25)),
                    "median": float(np.percentile(ranges, 50)),
                    "p75": float(np.percentile(ranges, 75)),
                    "max": float(np.max(ranges)),
                },
            }
            for name, smin, smax, cnt, ranges in subfolder_stats
        ],
        "global_range_distribution": global_range_stats,
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"    -> Scaling metadata saved to {metadata_path}")


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
    parser.add_argument("--gaussian-sigma", type=float, default=2.0,
                        help="Gaussian smoothing sigma per iteration (default: 2.0).")
    parser.add_argument("--smooth-iterations", type=int, default=3,
                        help="Number of smooth-then-rescale iterations (default: 3).")
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
        gaussian_iterations=args.smooth_iterations,
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
