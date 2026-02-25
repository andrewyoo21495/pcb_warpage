"""File I/O utilities: reading elevation data, saving preprocessed files, file discovery."""

import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EXCLUDE_SUFFIXES = ("ORI", "ORI@LOW", "ORI_A")


def should_skip_file(filepath: str, exclude_suffixes: list = EXCLUDE_SUFFIXES) -> bool:
    """Returns True if the filename (without extension) ends with any of the exclude_suffixes."""
    stem = Path(filepath).stem
    return any(stem.endswith(suffix) for suffix in exclude_suffixes)


def read_elevation(filepath: str, null_value: float = 9999.0) -> np.ndarray:
    """Load a tab-delimited elevation file and replace null sentinels with NaN.

    Args:
        filepath: Path to the .txt file.
        null_value: Sentinel value representing null (default 9999.0).

    Returns:
        np.ndarray of shape (N, M) with NaN for missing values.

    Raises:
        ValueError: If the file cannot be parsed.
    """
    data = np.loadtxt(filepath, delimiter='\t')
    logger.info("Loaded %s — shape: %s", filepath, data.shape)
    data[data == null_value] = np.nan
    return data


def save_preprocessed_txt(data: np.ndarray, output_path: str) -> None:
    """Save preprocessed data as tab-delimited text with 4 decimal places."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savetxt(output_path, data, delimiter='\t', fmt='%.4f')
    logger.info("Saved preprocessed text: %s", output_path)


def discover_subfolders(root_dir: str) -> List[str]:
    """Return a sorted list of immediate subdirectory paths under root_dir."""
    subfolders = []
    for entry in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, entry)
        if os.path.isdir(full_path):
            # Skip output directories
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
            logger.info("Skipping excluded file: %s", full_path)
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

    Args:
        filepath: Original input file path.
        parent_dir: Directory where interpolated/ and images/ will be created.

    Returns:
        (txt_output_path, img_output_path)
    """
    stem = Path(filepath).stem
    txt_path = os.path.join(parent_dir, "interpolated", f"{stem}_preprocessed.txt")
    img_path = os.path.join(parent_dir, "images", f"{stem}_preprocessed.png")
    return txt_path, img_path
