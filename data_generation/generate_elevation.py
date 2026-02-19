#!/usr/bin/env python3
"""Generate synthetic PCB elevation (warpage) images for each design variant.

Each elevation is a 256×256 grayscale image with values in [0, 1] (continuous,
smooth) representing surface height.

Generation formula:
  elevation = base_warp(design) + low_freq_noise() + small_random_tilt()
  elevation = clip(elevation, 0, 1)

Design-condition relationship:
  Higher line density region → slightly higher mean elevation
  (simulates thermal / mechanical stress concentration)

Per-sample variation:
  Sum of low-frequency sinusoid components + small random linear tilt.

Run:
  python -m data_generation.generate_elevation
  (reads data/design/*.png, writes data/elevation/design_{A,B,C,D}/)
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

CANVAS  = 256
SEED    = 42
N_SAMPLES = 200       # elevation images per design

DATA_DIR   = Path(__file__).parent.parent / 'data'
DESIGN_DIR = DATA_DIR / 'design'
ELEV_DIR   = DATA_DIR / 'elevation'

DESIGN_NAMES = ['design_A', 'design_B', 'design_C', 'design_D']

# Per-design warp scale (higher density → stronger base warp amplitude)
WARP_AMPLITUDE = {
    'design_A': 0.25,   # low density → mild warp
    'design_B': 0.35,   # medium density → moderate warp
    'design_C': 0.45,   # high density → strong warp
    'design_D': 0.38,   # medium density → tilted warp
}
# Per-design extra tilt magnitude (simulates asymmetric thermal stress)
TILT_SCALE = {
    'design_A': 0.05,
    'design_B': 0.10,
    'design_C': 0.05,
    'design_D': 0.15,   # top-heavy → stronger tilt
}


# ------------------------------------------------------------------
# Component functions
# ------------------------------------------------------------------

def _load_density_map(design_path: Path) -> np.ndarray:
    """Return normalised foreground density map (H, W) in [0, 1]."""
    img = np.array(Image.open(design_path).convert('L').resize(
        (CANVAS, CANVAS), Image.LANCZOS), dtype=np.float32)
    fg  = (img < 128).astype(np.float32)  # black pixels = foreground
    # Smooth with a large Gaussian to get a density map
    from scipy.ndimage import gaussian_filter
    density = gaussian_filter(fg, sigma=20)
    # Normalise to [0, 1]
    if density.max() > 0:
        density = density / density.max()
    return density


def _base_warp(density_map: np.ndarray, amplitude: float) -> np.ndarray:
    """Create a smooth base warp surface conditioned on design density.

    Combines:
      - A 2-D Gaussian bowl (simulates global PCB bending)
      - Scaled density map (high-density regions have higher elevation)
    """
    x = np.linspace(-1, 1, CANVAS)
    y = np.linspace(-1, 1, CANVAS)
    X, Y = np.meshgrid(x, y)

    # Gaussian bowl centred near middle-ish (with slight random offset per call)
    cx = np.random.uniform(-0.2, 0.2)
    cy = np.random.uniform(-0.2, 0.2)
    gaussian_bowl = np.exp(-((X - cx)**2 + (Y - cy)**2) / 0.6)

    warp = 0.5 * gaussian_bowl + 0.5 * density_map
    return warp * amplitude


def _low_freq_noise(num_components: int = 3, max_amplitude: float = 0.12) -> np.ndarray:
    """Sum of low-frequency sinusoidal waves — simulates manufacturing variability."""
    x = np.linspace(0, 2 * np.pi, CANVAS)
    y = np.linspace(0, 2 * np.pi, CANVAS)
    X, Y = np.meshgrid(x, y)

    noise = np.zeros((CANVAS, CANVAS), dtype=np.float32)
    for _ in range(num_components):
        fx    = np.random.uniform(0.3, 1.5)
        fy    = np.random.uniform(0.3, 1.5)
        px    = np.random.uniform(0, 2 * np.pi)
        py    = np.random.uniform(0, 2 * np.pi)
        amp   = np.random.uniform(0.01, max_amplitude)
        noise += amp * np.sin(fx * X + px) * np.sin(fy * Y + py)
    return noise


def _random_tilt(tilt_scale: float) -> np.ndarray:
    """Linear gradient in a random direction — simulates warped mounting."""
    x = np.linspace(0, 1, CANVAS)
    y = np.linspace(0, 1, CANVAS)
    X, Y = np.meshgrid(x, y)

    a = np.random.uniform(-tilt_scale, tilt_scale)
    b = np.random.uniform(-tilt_scale, tilt_scale)
    return (a * X + b * Y).astype(np.float32)


def _generate_single_elevation(
    density_map: np.ndarray,
    warp_amplitude: float,
    tilt_scale: float,
) -> np.ndarray:
    """Create one elevation sample in [0, 1]."""
    base   = _base_warp(density_map, warp_amplitude)
    noise  = _low_freq_noise()
    tilt   = _random_tilt(tilt_scale)
    elev   = base + noise + tilt
    # Shift so min ≈ 0, then clip
    elev   = elev - elev.min()
    if elev.max() > 0:
        elev = elev / elev.max()
    return elev.astype(np.float32)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def generate_all(
    design_dir: Path = DESIGN_DIR,
    elev_dir: Path   = ELEV_DIR,
    n_samples: int   = N_SAMPLES,
    seed: int        = SEED,
):
    np.random.seed(seed)
    random.seed(seed)

    elev_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {n_samples} elevation samples per design...")

    for design_name in DESIGN_NAMES:
        design_path = design_dir / f"{design_name}.png"
        if not design_path.exists():
            print(f"  [SKIP] {design_path} not found — run generate_design.py first.")
            continue

        out_dir = elev_dir / design_name
        out_dir.mkdir(parents=True, exist_ok=True)

        density_map    = _load_density_map(design_path)
        warp_amplitude = WARP_AMPLITUDE[design_name]
        tilt_scale     = TILT_SCALE[design_name]

        for i in range(n_samples):
            elev  = _generate_single_elevation(density_map, warp_amplitude, tilt_scale)
            # Convert float [0,1] → uint8 [0,255] for PNG storage
            img   = Image.fromarray((elev * 255).astype(np.uint8), mode='L')
            fname = out_dir / f"elevation_{i:04d}.png"
            img.save(fname)

        print(f"  {design_name}: {n_samples} samples → {out_dir}")

    print("Done.")


if __name__ == '__main__':
    generate_all()
