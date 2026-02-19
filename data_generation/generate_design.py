#!/usr/bin/env python3
"""Generate synthetic PCB design images (4 variants: A, B, C, D).

Each design is a 256×256 grayscale image:
  - White background (pixel = 255)
  - Black lines / pads / vias (pixel = 0)
  - Dilation applied after drawing to preserve thin line visibility

Target properties per variant:
  A — Low density (~5%),  Low asymmetry,         Dominant orientation: Horizontal
  B — Medium  (~15%), Medium asymmetry (left-heavy), Dominant: Mixed
  C — High    (~30%), Low asymmetry,             Dominant: Vertical
  D — Medium  (~20%), High asymmetry (top-heavy), Dominant: Mixed

Run:
  python -m data_generation.generate_design
  (creates data/design/design_{A,B,C,D}.png)
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

CANVAS = 256
OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'design'
SEED = 42


def _new_canvas() -> np.ndarray:
    """White 256×256 uint8 canvas."""
    return np.ones((CANVAS, CANVAS), dtype=np.uint8) * 255


def _draw_hlines(arr: np.ndarray, n: int, rng: np.random.Generator,
                 x_range: tuple = (0, CANVAS), y_range: tuple = (0, CANVAS)):
    """Draw horizontal line segments."""
    for _ in range(n):
        y = rng.integers(y_range[0], y_range[1])
        x0 = rng.integers(x_range[0], max(x_range[0] + 1, x_range[1] - 40))
        x1 = min(x0 + rng.integers(30, 80), x_range[1] - 1)
        thickness = rng.integers(1, 3)
        for dy in range(thickness):
            row = int(np.clip(y + dy, 0, CANVAS - 1))
            arr[row, x0:x1] = 0


def _draw_vlines(arr: np.ndarray, n: int, rng: np.random.Generator,
                 x_range: tuple = (0, CANVAS), y_range: tuple = (0, CANVAS)):
    """Draw vertical line segments."""
    for _ in range(n):
        x = rng.integers(x_range[0], x_range[1])
        y0 = rng.integers(y_range[0], max(y_range[0] + 1, y_range[1] - 40))
        y1 = min(y0 + rng.integers(30, 80), y_range[1] - 1)
        thickness = rng.integers(1, 3)
        for dx in range(thickness):
            col = int(np.clip(x + dx, 0, CANVAS - 1))
            arr[y0:y1, col] = 0


def _draw_rects(arr: np.ndarray, n: int, rng: np.random.Generator,
                x_range: tuple = (0, CANVAS), y_range: tuple = (0, CANVAS)):
    """Draw rectangular pad outlines."""
    for _ in range(n):
        x = rng.integers(x_range[0], x_range[1] - 20)
        y = rng.integers(y_range[0], y_range[1] - 20)
        w = rng.integers(6, 20)
        h = rng.integers(6, 20)
        x2, y2 = min(x + w, CANVAS - 1), min(y + h, CANVAS - 1)
        arr[y, x:x2] = 0
        arr[y2, x:x2] = 0
        arr[y:y2, x] = 0
        arr[y:y2, x2] = 0


def _draw_vias(arr: np.ndarray, n: int, rng: np.random.Generator,
               x_range: tuple = (0, CANVAS), y_range: tuple = (0, CANVAS)):
    """Draw small circular via markers."""
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    for _ in range(n):
        cx = rng.integers(x_range[0] + 4, x_range[1] - 4)
        cy = rng.integers(y_range[0] + 4, y_range[1] - 4)
        r = rng.integers(2, 5)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=1)
    return np.array(img)


def _dilate(arr: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply morphological dilation to preserve thin lines after possible resize."""
    fg = (arr < 128)  # foreground = dark pixels
    dilated = binary_dilation(fg, iterations=iterations)
    result = arr.copy()
    result[dilated] = 0
    return result


# ------------------------------------------------------------------
# Design variant generators
# ------------------------------------------------------------------

def generate_design_A(rng: np.random.Generator) -> np.ndarray:
    """Low density (~5%), Low asymmetry, Horizontal dominant."""
    arr = _new_canvas()
    _draw_hlines(arr, n=8,  rng=rng)
    _draw_vlines(arr, n=2,  rng=rng)
    _draw_rects (arr, n=3,  rng=rng)
    arr = _draw_vias(arr, n=4, rng=rng)
    return _dilate(arr)


def generate_design_B(rng: np.random.Generator) -> np.ndarray:
    """Medium density (~15%), Medium left-heavy asymmetry, Mixed orientation."""
    arr = _new_canvas()
    # More elements on the left half
    _draw_hlines(arr, n=8,  rng=rng, x_range=(0, 128))
    _draw_vlines(arr, n=6,  rng=rng, x_range=(0, 128))
    _draw_rects (arr, n=6,  rng=rng, x_range=(0, 128))
    # Fewer on the right
    _draw_hlines(arr, n=3,  rng=rng, x_range=(128, 256))
    _draw_vlines(arr, n=2,  rng=rng, x_range=(128, 256))
    _draw_rects (arr, n=2,  rng=rng, x_range=(128, 256))
    arr = _draw_vias(arr, n=8, rng=rng)
    return _dilate(arr)


def generate_design_C(rng: np.random.Generator) -> np.ndarray:
    """High density (~30%), Low asymmetry, Vertical dominant."""
    arr = _new_canvas()
    _draw_vlines(arr, n=30, rng=rng)
    _draw_hlines(arr, n=6,  rng=rng)
    _draw_rects (arr, n=10, rng=rng)
    arr = _draw_vias(arr, n=10, rng=rng)
    return _dilate(arr)


def generate_design_D(rng: np.random.Generator) -> np.ndarray:
    """Medium density (~20%), High top-heavy asymmetry, Mixed orientation."""
    arr = _new_canvas()
    # Concentrated in top half
    _draw_hlines(arr, n=10, rng=rng, y_range=(0, 128))
    _draw_vlines(arr, n=8,  rng=rng, y_range=(0, 128))
    _draw_rects (arr, n=8,  rng=rng, y_range=(0, 128))
    # Sparse in bottom half
    _draw_hlines(arr, n=2,  rng=rng, y_range=(128, 256))
    _draw_vlines(arr, n=2,  rng=rng, y_range=(128, 256))
    arr = _draw_vias(arr, n=6, rng=rng, y_range=(0, 128))
    return _dilate(arr)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

GENERATORS = {
    'A': generate_design_A,
    'B': generate_design_B,
    'C': generate_design_C,
    'D': generate_design_D,
}


def generate_all(output_dir: Path = OUTPUT_DIR, seed: int = SEED):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("Generating synthetic design images...")
    for name, gen_fn in GENERATORS.items():
        arr   = gen_fn(rng)
        path  = output_dir / f"design_{name}.png"
        Image.fromarray(arr.astype(np.uint8), mode='L').save(path)

        density = (arr < 128).mean() * 100
        print(f"  design_{name}.png  —  line density: {density:.1f}%  →  {path}")

    print("Done.")


if __name__ == '__main__':
    random.seed(SEED)
    generate_all()
