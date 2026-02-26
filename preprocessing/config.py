"""Configuration dataclass and defaults for PCB Elevation Preprocessor."""

from dataclasses import dataclass, field
from typing import List, Optional


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

    # Tilt correction
    tilt_patch_size: int = 16

    # Gaussian smoothing
    gaussian_sigma: float = 2.0
    gaussian_iterations: int = 3

    # Parallel processing
    max_workers: int = 1

    # Output
    image_format: str = "png"
    colormap: str = "gray"
