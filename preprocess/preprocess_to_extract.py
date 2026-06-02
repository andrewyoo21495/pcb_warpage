"""
PCB Warpage: Preprocessing + Feature Extraction Pipeline
=========================================================
Runs the full pipeline from raw elevation data to feature Excel output:
  1. Preprocess (downsample, tilt correction, outlier removal, interpolation, smoothing)
  2. Save preprocessed .txt and grayscale image
  3. Extract 12 warpage feature variables
  4. Export features to Excel

Requirements:
    pip install numpy scipy scikit-learn Pillow openpyxl

Usage:
    # Single file
    python preprocess/preprocess_to_extract.py --input-file /path/to/raw.txt

    # Directory (all .txt files in folder)
    python preprocess/preprocess_to_extract.py --input-dir /path/to/folder

    # With custom preprocessing parameters
    python preprocess/preprocess_to_extract.py --input-dir /path/to/folder \
        --downsample-factor 4 --z-threshold 3.0 --poly-degree 3

    # Custom output Excel path
    python preprocess/preprocess_to_extract.py --input-dir /path/to/folder \
        --output features.xlsx
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Import preprocessing functions from preprocess_total
from preprocess_total import (
    PreprocessorConfig,
    process_single_file,
    save_preprocessed_txt,
    generate_grayscale_image,
    should_skip_file,
    EXCLUDE_SUFFIXES,
)

# Import feature extraction and Excel export from wap_parameter_extract
from wap_parameter_extract import (
    extract_all_features,
    export_to_excel,
    FEATURE_COLUMNS,
)


# =============================================================================
# File Discovery
# =============================================================================

def discover_txt_files(directory: str, exclude_suffixes: list) -> list:
    """Find all .txt files in a directory, excluding skip-suffixes and output dirs."""
    txt_files = []
    for entry in sorted(os.listdir(directory)):
        if not entry.lower().endswith('.txt'):
            continue
        full_path = os.path.join(directory, entry)
        if not os.path.isfile(full_path):
            continue
        if should_skip_file(full_path, exclude_suffixes):
            continue
        txt_files.append(full_path)
    return txt_files


# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline(filepaths: list, config: PreprocessorConfig,
                 output_xlsx: str) -> None:
    """Run preprocessing + feature extraction on a list of files."""
    total = len(filepaths)
    results = []
    files_ok = 0
    files_fail = 0

    print(f"\n  Processing {total} file(s)...\n")
    print(f"  {'#':>4s}  {'File':<40s}  {'Status':<8s}  {'Max':>8s}  {'Mode':<8s}")
    print(f"  {'':->4s}  {'':->40s}  {'':->8s}  {'':->8s}  {'':->8s}")

    for i, filepath in enumerate(filepaths, start=1):
        stem = Path(filepath).stem
        parent_dir = os.path.dirname(filepath)
        display_name = stem if len(stem) <= 40 else stem[:37] + "..."

        # --- Phase 1: Preprocess ---
        data, n_outliers, n_interp, plane_amp = process_single_file(filepath, config)

        if data is None:
            print(f"  {i:4d}  {display_name:<40s}  SKIP")
            files_fail += 1
            continue

        # Save preprocessed txt
        txt_out = os.path.join(parent_dir, "interpolated", f"{stem}_preprocessed.txt")
        save_preprocessed_txt(data, txt_out)

        # Save grayscale image
        img_out = os.path.join(parent_dir, "images", f"{stem}_preprocessed.png")
        generate_grayscale_image(data, config.scale_min, config.scale_max, img_out)

        # --- Phase 2: Extract features ---
        features = extract_all_features(data)
        record = {"pcb_name": stem}
        record.update(features)
        results.append(record)

        mode = features.get("warpage_mode", "?")
        max_w = features.get("max_warpage", 0)
        print(f"  {i:4d}  {display_name:<40s}  OK        {max_w:8.1f}  {mode:<8s}")
        files_ok += 1

    # --- Summary ---
    print(f"\n  {'':->70s}")
    print(f"  Preprocessed: {files_ok} OK, {files_fail} skipped")

    if not results:
        print("  No files were successfully processed. No Excel output.")
        return

    # --- Phase 3: Export to Excel ---
    print(f"  Exporting {len(results)} records to: {output_xlsx}")
    export_to_excel(results, output_xlsx)
    print(f"  Excel saved: {output_xlsx}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PCB Warpage: Preprocessing + Feature Extraction Pipeline"
    )

    # Input mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-file", type=str,
                       help="Single raw .txt file to process.")
    group.add_argument("--input-dir", type=str,
                       help="Directory containing raw .txt files.")

    # Preprocessing parameters
    parser.add_argument("--downsample-factor", type=int, default=4)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=0.1)
    parser.add_argument("--tilt-patch-size", type=int, default=16)
    parser.add_argument("--gaussian-sigma", type=float, default=2.0)
    parser.add_argument("--smooth-iterations", type=int, default=3)

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output Excel file path (default: warpage_features.xlsx in input location).")

    args = parser.parse_args()

    # Build config
    config = PreprocessorConfig(
        downsample_factor=args.downsample_factor,
        outlier_z_threshold=args.z_threshold,
        outlier_grid_size=args.grid_size,
        interp_poly_degree=args.poly_degree,
        interp_ridge_alpha=args.ridge_alpha,
        tilt_patch_size=args.tilt_patch_size,
        gaussian_sigma=args.gaussian_sigma,
        gaussian_iterations=args.smooth_iterations,
    )

    start_time = time.time()
    print("\n" + "=" * 70)
    print("  PCB Warpage: Preprocessing + Feature Extraction Pipeline")
    print("=" * 70)

    # Discover files
    if args.input_file:
        if not os.path.isfile(args.input_file):
            print(f"  [ERROR] File not found: {args.input_file}")
            sys.exit(1)
        filepaths = [args.input_file]
        default_output_dir = os.path.dirname(args.input_file) or "."
        print(f"  Mode: Single file")
        print(f"  Input: {args.input_file}")
    else:
        if not os.path.isdir(args.input_dir):
            print(f"  [ERROR] Directory not found: {args.input_dir}")
            sys.exit(1)
        filepaths = discover_txt_files(args.input_dir, config.exclude_suffixes)
        default_output_dir = args.input_dir
        print(f"  Mode: Directory ({len(filepaths)} .txt files found)")
        print(f"  Input: {args.input_dir}")

    if not filepaths:
        print("  No .txt files found. Nothing to do.")
        sys.exit(0)

    output_xlsx = args.output or os.path.join(default_output_dir, "warpage_features.xlsx")
    print(f"  Output: {output_xlsx}")

    # Run pipeline
    run_pipeline(filepaths, config, output_xlsx)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  Completed in {elapsed:.1f} seconds.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
