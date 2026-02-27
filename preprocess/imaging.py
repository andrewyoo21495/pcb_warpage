"""Global min/max computation, min-max scaling, and grayscale image generation."""

import logging
import os
from typing import Tuple

import numpy as np
from PIL import Image

from .io_utils import discover_preprocessed_files

logger = logging.getLogger(__name__)


def compute_global_minmax(root_dir: str) -> Tuple[float, float]:
    """Compute global min and max across all preprocessed files.

    Args:
        root_dir: Root directory containing subfolders with interpolated/ dirs.

    Returns:
        (global_min, global_max) across all preprocessed files.
    """
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
    """Scale data to [0, 255] using global min/max and save as grayscale image.

    Args:
        data: 2D array of preprocessed elevation values.
        global_min: Global minimum for scaling.
        global_max: Global maximum for scaling.
        output_path: Path to save the output image.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if global_min == global_max:
        logger.warning("global_min == global_max (%.4f). Generating mid-value image.",
                        global_min)
        img_array = np.full(data.shape, 128, dtype=np.uint8)
    else:
        scaled = (data - global_min) / (global_max - global_min)
        scaled = np.clip(scaled, 0.0, 1.0)
        img_array = (scaled * 255).astype(np.uint8)

    img = Image.fromarray(img_array, mode='L')
    img.save(output_path)
