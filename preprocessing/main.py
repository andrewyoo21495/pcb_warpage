"""CLI entry point and orchestration for PCB Elevation Preprocessor."""

import argparse
import logging
import sys

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
        logger.error("Failed to read %s: %s", filepath, e)
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


def run_single_file_mode(config: PreprocessorConfig) -> None:
    """Single-file mode: process one file and generate output."""
    import os

    filepath = config.input_file
    parent_dir = os.path.dirname(filepath)
    txt_path, img_path = get_output_paths(filepath, parent_dir)

    data = process_single_file(filepath, config)
    if data is None:
        logger.error("Processing failed for %s", filepath)
        return

    save_preprocessed_txt(data, txt_path)

    # Single file: use its own range for scaling
    global_min, global_max = float(np.min(data)), float(np.max(data))
    logger.info("Single-file global min/max: min=%.4f, max=%.4f",
                global_min, global_max)
    generate_grayscale_image(data, global_min, global_max, img_path)


def run_batch_mode(config: PreprocessorConfig) -> None:
    """Batch (directory) mode: process all subfolders, then generate images."""
    import os

    root_dir = config.root_dir
    subfolders = discover_subfolders(root_dir)

    if not subfolders:
        logger.warning("No subfolders found in %s", root_dir)
        return

    # Phase 1: Per-file preprocessing (Steps 1-5)
    logger.info("=== Phase 1: Preprocessing ===")
    files_processed = 0

    for subfolder in subfolders:
        txt_files = discover_txt_files(subfolder, config.exclude_suffixes)
        if not txt_files:
            logger.info("No .txt files in %s — skipping.", subfolder)
            continue

        for filepath in txt_files:
            logger.info("Processing: %s", filepath)
            data = process_single_file(filepath, config)
            if data is None:
                continue

            txt_path, _ = get_output_paths(filepath, subfolder)
            save_preprocessed_txt(data, txt_path)
            files_processed += 1

    logger.info("Phase 1 complete: %d files preprocessed.", files_processed)

    if files_processed == 0:
        logger.warning("No files were preprocessed. Skipping Phase 2.")
        return

    # Phase 2: Global scaling & image generation (Steps 6-7)
    logger.info("=== Phase 2: Image Generation ===")
    global_min, global_max = compute_global_minmax(root_dir)

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

    logger.info("Phase 2 complete.")


def main(config: PreprocessorConfig) -> None:
    """Main orchestration: dispatch to single-file or batch mode."""
    if config.input_file and config.root_dir:
        raise ValueError("Specify either --root-dir or --input-file, not both.")
    if not config.input_file and not config.root_dir:
        raise ValueError("Specify either --root-dir or --input-file.")

    if config.input_file:
        run_single_file_mode(config)
    else:
        run_batch_mode(config)


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
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = parse_args()
    try:
        main(config)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)
