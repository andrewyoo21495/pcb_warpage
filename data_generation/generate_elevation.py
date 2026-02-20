#!/usr/bin/env python3
"""Generate synthetic PCB elevation (warpage) images for each design variant.

Each elevation is a 256×256 grayscale image with values in [0, 1] (continuous,
smooth) representing surface height.

Pattern types (randomly chosen per sample):
  center_bump     — centre area is highest (convex bump)
  center_bowl     — centre area is lowest  (concave bowl / valley)
  corner_single   — one random corner is elevated or depressed
  corner_diagonal — two diagonal corners are elevated or depressed (saddle)
  corner_adjacent — two adjacent corners are elevated or depressed (ramp)
  corner_all      — all four corners are elevated or depressed

Generation formula (per sample):
  elevation = pattern_surface * amplitude
            + density_map * 0.12          (weak design conditioning)
            + low_freq_noise()
            + random_tilt()
  elevation = normalise(elevation) to [0, 1]

Run:
  python -m data_generation.generate_elevation
  (reads data/design/*.png, writes data/elevation/design_{A,...,J}/)
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

CANVAS    = 256
SEED      = 42
N_SAMPLES = 300       # elevation images per design

DATA_DIR   = Path(__file__).parent.parent / 'data'
DESIGN_DIR = DATA_DIR / 'design'
ELEV_DIR   = DATA_DIR / 'elevation'

DESIGN_NAMES = [
    'design_A', 'design_B', 'design_C', 'design_D',
    'design_E', 'design_F', 'design_G', 'design_H', 'design_I', 'design_J',
]

# Per-design warp amplitude: scales the dominant pattern surface
WARP_AMPLITUDE = {
    'design_A': 0.25,
    'design_B': 0.35,
    'design_C': 0.45,
    'design_D': 0.38,
    'design_E': 0.30,
    'design_F': 0.28,
    'design_G': 0.40,
    'design_H': 0.35,
    'design_I': 0.32,
    'design_J': 0.42,
}
# Per-design tilt magnitude: simulates asymmetric thermal / mounting stress
TILT_SCALE = {
    'design_A': 0.05,
    'design_B': 0.10,
    'design_C': 0.05,
    'design_D': 0.15,
    'design_E': 0.08,
    'design_F': 0.06,
    'design_G': 0.10,
    'design_H': 0.12,
    'design_I': 0.07,
    'design_J': 0.15,
}

# Ordered list of pattern types — sampled uniformly at random per elevation
PATTERN_TYPES = [
    'center_bump',      # centre is the peak
    'center_bowl',      # centre is the valley
    'corner_single',    # one corner is high or low
    'corner_diagonal',  # two diagonal corners are high or low
    'corner_adjacent',  # two adjacent corners are high or low
    'corner_all',       # all four corners are high or low
]

# Corner positions in the (−1, 1)² coordinate space, going clockwise from TL
_CORNERS = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
_DIAGONAL_PAIRS = [
    [(-1.0, -1.0), ( 1.0,  1.0)],   # top-left / bottom-right
    [( 1.0, -1.0), (-1.0,  1.0)],   # top-right / bottom-left
]
_ADJACENT_PAIRS = [
    [(-1.0, -1.0), ( 1.0, -1.0)],   # top edge
    [( 1.0, -1.0), ( 1.0,  1.0)],   # right edge
    [( 1.0,  1.0), (-1.0,  1.0)],   # bottom edge
    [(-1.0,  1.0), (-1.0, -1.0)],   # left edge
]


# ------------------------------------------------------------------
# Component functions
# ------------------------------------------------------------------

def _load_density_map(design_path: Path) -> np.ndarray:
    """Return a normalised foreground-density map (H, W) in [0, 1]."""
    img = np.array(Image.open(design_path).convert('L').resize(
        (CANVAS, CANVAS), Image.LANCZOS), dtype=np.float32)
    fg = (img < 128).astype(np.float32)   # black pixels = foreground
    from scipy.ndimage import gaussian_filter
    density = gaussian_filter(fg, sigma=20)
    if density.max() > 0:
        density = density / density.max()
    return density


def _pattern_surface(pattern_type: str) -> np.ndarray:
    """Return a (CANVAS, CANVAS) float32 base surface for *pattern_type*.

    All patterns are built from sums of axis-aligned Gaussian bells so the
    surface is always smooth.  A random sign-flip (high ↔ low) is applied
    with 50 % probability to every corner-weighted variant, and both
    center_bump / center_bowl are separate explicit types.
    """
    x = np.linspace(-1.0, 1.0, CANVAS)
    y = np.linspace(-1.0, 1.0, CANVAS)
    X, Y = np.meshgrid(x, y)

    def _gauss(cx: float, cy: float, sigma: float) -> np.ndarray:
        return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (sigma ** 2))

    if pattern_type == 'center_bump':
        # Gaussian centred near the middle — centre is the peak
        cx    = np.random.uniform(-0.15, 0.15)
        cy    = np.random.uniform(-0.15, 0.15)
        sigma = np.random.uniform(0.35, 0.80)
        z     = _gauss(cx, cy, sigma)

    elif pattern_type == 'center_bowl':
        # Inverted Gaussian — centre is the valley, edges are high
        cx    = np.random.uniform(-0.15, 0.15)
        cy    = np.random.uniform(-0.15, 0.15)
        sigma = np.random.uniform(0.35, 0.80)
        z     = 1.0 - _gauss(cx, cy, sigma)

    elif pattern_type == 'corner_single':
        # Gaussian placed at one randomly chosen corner
        cx, cy = _CORNERS[np.random.randint(4)]
        sigma  = np.random.uniform(0.60, 1.40)
        z      = _gauss(cx, cy, sigma)
        if np.random.random() < 0.5:          # 50 %: that corner is LOW
            z = 1.0 - z

    elif pattern_type == 'corner_diagonal':
        # Two diagonal corners share the same elevation (high or low)
        pair  = _DIAGONAL_PAIRS[np.random.randint(2)]
        sigma = np.random.uniform(0.50, 1.20)
        z     = sum(_gauss(cx, cy, sigma) for cx, cy in pair)
        if np.random.random() < 0.5:          # 50 %: diagonal corners LOW
            z = float(z.max()) - z

    elif pattern_type == 'corner_adjacent':
        # Two corners sharing an edge are elevated or depressed together
        pair  = _ADJACENT_PAIRS[np.random.randint(4)]
        sigma = np.random.uniform(0.50, 1.20)
        z     = sum(_gauss(cx, cy, sigma) for cx, cy in pair)
        if np.random.random() < 0.5:          # 50 %: adjacent corners LOW
            z = float(z.max()) - z

    elif pattern_type == 'corner_all':
        # All four corners share the same elevation extreme
        sigma = np.random.uniform(0.45, 1.00)
        z     = sum(_gauss(cx, cy, sigma) for cx, cy in _CORNERS)
        if np.random.random() < 0.5:          # 50 %: all corners LOW, centre HIGH
            z = float(z.max()) - z

    else:
        raise ValueError(f"Unknown pattern type: {pattern_type!r}")

    return z.astype(np.float32)


def _low_freq_noise(num_components: int = 3, max_amplitude: float = 0.12) -> np.ndarray:
    """Sum of low-frequency sinusoidal waves — simulates manufacturing variability."""
    x = np.linspace(0, 2 * np.pi, CANVAS)
    y = np.linspace(0, 2 * np.pi, CANVAS)
    X, Y = np.meshgrid(x, y)

    noise = np.zeros((CANVAS, CANVAS), dtype=np.float32)
    for _ in range(num_components):
        fx  = np.random.uniform(0.3, 1.5)
        fy  = np.random.uniform(0.3, 1.5)
        px  = np.random.uniform(0, 2 * np.pi)
        py  = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.01, max_amplitude)
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
    """Create one elevation sample in [0, 1].

    A pattern type is chosen at random for each sample; the density map adds
    a weak design-specific bias so the CVAE can learn design conditioning.
    """
    pattern_type = np.random.choice(PATTERN_TYPES)

    pattern = _pattern_surface(pattern_type) * warp_amplitude
    density_bias = density_map * 0.12
    noise = _low_freq_noise()
    tilt  = _random_tilt(tilt_scale)

    elev = pattern + density_bias + noise + tilt
    elev = elev - elev.min()
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
        tilt_scale_val = TILT_SCALE[design_name]

        for i in range(n_samples):
            elev  = _generate_single_elevation(density_map, warp_amplitude, tilt_scale_val)
            img   = Image.fromarray((elev * 255).astype(np.uint8), mode='L')
            fname = out_dir / f"elevation_{i:04d}.png"
            img.save(fname)

        print(f"  {design_name}: {n_samples} samples → {out_dir}")

    print("Done.")


if __name__ == '__main__':
    generate_all()
