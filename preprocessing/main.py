"""CLI entry point and orchestration for PCB Elevation Preprocessor."""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import PreprocessorConfig
from .imaging import compute_global_minmax, generate_grayscale_image
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


def process_single_file(filepath: str, config: PreprocessorConfig) -> np.ndarray:
    """Run Steps 1-5 on a single file and return the preprocessed data.

    Returns:
        Preprocessed numpy array, or None if the file should be skipped.
    """
    try:
        data = read_elevation(filepath, null_value=config.null_value)
    except Exception as e:
        logger.warning("Failed to read %s: %s", filepath, e)
        return None

    # Check if entire data is NaN
    if np.all(np.isnan(data)):
        logger.warning("Entire data is NaN in %s — skipping.", filepath)
        return None

    data = downsample_median(data, factor=config.downsample_factor)
    data = detect_and_remove_outliers(
        data,
        grid_size=config.outlier_grid_size,
        z_threshold=config.outlier_z_threshold,
    )
    data = interpolate_surface(
        data,
        poly_degree=config.interp_poly_degree,
        ridge_alpha=config.interp_ridge_alpha,
    )
    data = smooth_gaussian(data, sigma=config.gaussian_sigma)

    return data


def _process_and_save(filepath: str, output_dir: str, config: PreprocessorConfig) -> str:
    """Process a single file and save outputs. Returns filepath on success, None on failure.

    This is the worker function used by both sequential and parallel modes.
    """
    data = process_single_file(filepath, config)
    if data is None:
        return None

    txt_path, _ = get_output_paths(filepath, output_dir)
    save_preprocessed_txt(data, txt_path)
    return filepath


def run_single_file_mode(config: PreprocessorConfig) -> None:
    """Single-file mode: process one file and generate output."""
    filepath = config.input_file
    parent_dir = os.path.dirname(filepath)
    txt_path, img_path = get_output_paths(filepath, parent_dir)

    print(f"  Processing: {Path(filepath).name}")
    data = process_single_file(filepath, config)
    if data is None:
        print(f"  [FAILED] {filepath}")
        return

    save_preprocessed_txt(data, txt_path)

    global_min, global_max = float(np.min(data)), float(np.max(data))
    generate_grayscale_image(data, global_min, global_max, img_path)
    print(f"  [DONE] Saved to: {txt_path}")
    print(f"         Image:    {img_path}")


def run_batch_mode(config: PreprocessorConfig) -> None:
    """Batch (directory) mode: process all subfolders, then generate images."""
    root_dir = config.root_dir
    subfolders = discover_subfolders(root_dir)

    if not subfolders:
        print("  No subfolders found. Nothing to do.")
        return

    # Collect all files to process
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
                    result = fut.result()
                    if result is not None:
                        files_processed += 1
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
            result = _process_and_save(filepath, subfolder, config)
            if result is not None:
                files_processed += 1
            else:
                files_skipped += 1

            done = files_processed + files_skipped
            print(f"\r    [{done}/{total_files}] processed", end="", flush=True)

    print()  # newline after progress
    print(f"    -> {files_processed} succeeded, {files_skipped} skipped.\n")

    if files_processed == 0:
        print("  No files were preprocessed. Skipping image generation.")
        return

    # --- Phase 2: Global scaling & image generation (Steps 6-7) ---
    print("  Phase 2/2: Generating images...")
    global_min, global_max = compute_global_minmax(root_dir)
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
