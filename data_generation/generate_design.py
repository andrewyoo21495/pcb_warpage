#!/usr/bin/env python3
"""Generate synthetic PCB design images (4 variants: A, B, C, D).

Each 256×256 grayscale image contains multiple copies of the same complex PCB
polygon footprint placed symmetrically on a white background.  Every footprint
is drawn as an outline-only polygon (no fill) with intricate, winding shapes.

Variant  Shape                           Layout
  A      Complex DIP + pin slots         2×2 symmetric grid (4 copies)
  B      Winding staircase               3×2 symmetric grid (6 copies)
  C      Notched cross                   2×2 symmetric grid (4 copies)
  D      Notched chamfered octagon       2×3 symmetric grid (6 copies)
  E      Ascending staircase band        2×2 symmetric grid (4 copies)
  F      T-shape + pin-slot notches      2×2 symmetric grid (4 copies)
  G      Hourglass with notched ends     2×2 symmetric grid (4 copies)
  H      Three-tooth comb                2×2 symmetric grid (4 copies)
  I      I-beam with notched flanges     2×2 symmetric grid (4 copies)
  J      U-channel with wall notches     2×2 symmetric grid (4 copies)

Run:
  python -m data_generation.generate_design
  (creates data/design/design_{A,B,C,D,E,F,G,H,I,J}.png)
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CANVAS   = 256
BORDER   = 4       # minimum pixels between grid edge and image edge
GAP      = 8       # gap between two polygon bounding boxes
LINE_W   = 2       # outline stroke width in pixels

OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'design'
SEED       = 42


# ---------------------------------------------------------------------------
# Polygon templates (complex, winding outlines)
# Each function returns a list of (x, y) vertices relative to bbox top-left.
# All coordinates are non-negative; bounding box min is at (0, 0).
# ---------------------------------------------------------------------------

def _template_A():
    """Complex DIP outline — 40×40 bbox, pin-slot notches on all four sides.

    The shape is a rectangle with:
      - A centre notch on the top edge (IC orientation marker)
      - Two inward rectangular pin-slot notches on each vertical side
      - Two inward rectangular notches on the bottom edge
    """
    return [
        # Start at top-left of the top notch, tracing clockwise
        (0, 8),
        # Left side: 2 inward pin-slot notches
        (0, 14), (5, 14), (5, 18), (0, 18),
        (0, 24), (5, 24), (5, 28), (0, 28),
        # Bottom-left corner
        (0, 40),
        # Bottom edge: 2 inward notches
        (10, 40), (10, 35), (16, 35), (16, 40),
        (24, 40), (24, 35), (30, 35), (30, 40),
        # Bottom-right corner
        (40, 40),
        # Right side: 2 inward pin-slot notches
        (40, 28), (35, 28), (35, 24), (40, 24),
        (40, 18), (35, 18), (35, 14), (40, 14),
        # Top-right
        (40, 8),
        # Top edge: centre notch (orientation marker)
        (26, 8), (26, 0), (14, 0), (14, 8),
    ]


def _template_B():
    """Winding staircase polygon — 44×38 bbox, four concave notches.

    The boundary steps in/out at the top, right, bottom, and left, giving
    a complex, winding outline with no straight monotone sides.
    """
    return [
        # Stepped top: notch between x=20 and x=28
        (0, 0),  (20, 0),
        (20, 8), (28, 8),
        (28, 0), (44, 0),
        # Right side: inward notch between y=14 and y=22
        (44, 14),
        (36, 14), (36, 22),
        (44, 22),
        # Stepped bottom: notch between x=18 and x=26
        (44, 38), (26, 38),
        (26, 30), (18, 30),
        (18, 38), (0, 38),
        # Left side: inward notch between y=16 and y=24
        (0, 24), (8, 24),
        (8, 16), (0, 16),
    ]


def _template_C():
    """Complex cross — 36×36 bbox, each arm tip carries 2 rectangular slots.

    Based on the original plus/cross but every arm end has two rectangular
    slots cut into it, making the outline highly winding (44 vertices total).
    """
    return [
        # Top arm: 2 slots cut into the tip (pointing upward)
        (10, 0), (14, 0), (14, 6), (16, 6), (16, 0),
        (20, 0), (20, 6), (22, 6), (22, 0), (26, 0),
        # Top-right inner corner
        (26, 10), (36, 10),
        # Right arm: 2 slots (pointing rightward)
        (36, 14), (30, 14), (30, 16), (36, 16),
        (36, 20), (30, 20), (30, 22), (36, 22), (36, 26),
        # Bottom-right inner corner
        (26, 26), (26, 36),
        # Bottom arm: 2 slots (pointing downward)
        (22, 36), (22, 30), (20, 30), (20, 36),
        (16, 36), (16, 30), (14, 30), (14, 36), (10, 36),
        # Bottom-left inner corner
        (10, 26), (0, 26),
        # Left arm: 2 slots (pointing leftward)
        (0, 22), (6, 22), (6, 20), (0, 20),
        (0, 16), (6, 16), (6, 14), (0, 14), (0, 10),
        # Top-left inner corner — PIL auto-closes back to (10, 0)
        (10, 10),
    ]


def _template_D():
    """Notched chamfered octagon — 40×40 bbox, inward notch on each flat edge.

    The chamfered (45°-corner) octagon has a rectangular slot cut into
    the centre of each of its four axis-aligned flat edges, giving a
    24-vertex winding outline.
    """
    return [
        # Top flat edge with centre notch
        (10, 0), (18, 0), (18, 6), (22, 6), (22, 0), (30, 0),
        # Top-right chamfer (diagonal)
        (40, 10),
        # Right flat edge with centre notch
        (40, 18), (34, 18), (34, 22), (40, 22), (40, 30),
        # Bottom-right chamfer
        (30, 40),
        # Bottom flat edge with centre notch
        (22, 40), (22, 34), (18, 34), (18, 40), (10, 40),
        # Bottom-left chamfer
        (0, 30),
        # Left flat edge with centre notch
        (0, 22), (6, 22), (6, 18), (0, 18), (0, 10),
        # Top-left chamfer — PIL auto-closes back to (10, 0)
    ]


def _template_E():
    """Ascending diagonal staircase band — 40×40 bbox.

    The outer edge climbs from bottom-left to top-right in 3 steps; the inner
    edge mirrors it 8 px inward, creating a diagonal strip outline.
    """
    return [
        # Outer staircase: bottom-left → top-right
        (0, 30), (10, 30), (10, 20), (20, 20),
        (20, 10), (30, 10), (30,  0), (40,  0),
        # Inner staircase: top-right → bottom-left
        (40, 10), (32, 10), (32, 20),
        (24, 20), (24, 30), (16, 30),
        (16, 40), ( 0, 40),
    ]


def _template_F():
    """T-shape with pin-slot notches — 50×44 bbox.

    A wide horizontal bar on top narrows to a vertical stem; the stem has
    two outward rectangular pin slots on each side.
    """
    return [
        # Horizontal bar
        ( 0,  0), (50,  0), (50, 16),
        # Right inner step to stem
        (36, 16),
        # Stem right side: 2 outward pin-slot notches
        (36, 22), (44, 22), (44, 28), (36, 28),
        (36, 34), (44, 34), (44, 40), (36, 40),
        (36, 44),
        # Stem bottom edge
        (14, 44),
        # Stem left side: 2 outward pin-slot notches (mirror)
        (14, 40), ( 6, 40), ( 6, 34), (14, 34),
        (14, 28), ( 6, 28), ( 6, 22), (14, 22),
        # Left inner step back to bar
        (14, 16), ( 0, 16),
    ]


def _template_G():
    """Hourglass outline — 44×36 bbox, diagonal waist with notched ends.

    Wide at top and bottom; two diagonal edges converge to a narrow waist
    in the middle.  Both flat ends carry 2 inward slots.
    """
    return [
        # Top edge with 2 inward slots
        ( 0,  0), (16,  0), (16,  6), (20,  6), (20,  0),
        (24,  0), (24,  6), (28,  6), (28,  0), (44,  0),
        # Right diagonal down to waist then back out
        (44, 14), (28, 18), (44, 22),
        (44, 36),
        # Bottom edge with 2 inward slots
        (28, 36), (28, 30), (24, 30), (24, 36),
        (20, 36), (20, 30), (16, 30), (16, 36),
        ( 0, 36),
        # Left diagonal up to waist then back to top
        ( 0, 22), (16, 18), ( 0, 14),
    ]


def _template_H():
    """Three-tooth comb — 40×48 bbox.

    A horizontal top bar (with 2 inward slots on its top edge) sprouts
    three evenly spaced downward teeth separated by open gaps.
    """
    return [
        # Top bar with 2 inward slots
        ( 0,  0), (12,  0), (12,  6), (16,  6), (16,  0),
        (24,  0), (24,  6), (28,  6), (28,  0), (40,  0),
        # Right tooth
        (40, 48), (30, 48), (30, 10),
        # Gap then middle tooth
        (25, 10), (25, 48), (15, 48), (15, 10),
        # Gap then left tooth
        (10, 10), (10, 48), ( 0, 48),
    ]


def _template_I():
    """I-beam outline — 40×40 bbox, both flanges have 2 inward notches.

    Top and bottom flanges span the full width; a narrow web connects them
    in the centre.
    """
    return [
        # Top flange with 2 inward notches
        ( 0,  0), (14,  0), (14,  6), (18,  6), (18,  0),
        (22,  0), (22,  6), (26,  6), (26,  0), (40,  0),
        # Right side of top flange → step in to web
        (40, 12), (26, 12),
        # Web right side
        (26, 28),
        # Step out to bottom flange → right side going down
        (40, 28), (40, 40),
        # Bottom flange with 2 inward notches
        (26, 40), (26, 34), (22, 34), (22, 40),
        (18, 40), (18, 34), (14, 34), (14, 40),
        # Left side of bottom flange → step in to web
        ( 0, 40), ( 0, 28), (14, 28),
        # Web left side
        (14, 12),
        # Step out to top flange
        ( 0, 12),
    ]


def _template_J():
    """U-channel outline — 36×40 bbox, open at the top, notched outer walls.

    Solid base with upward notches; left and right arms carry inward notches
    on their outer faces.  The inner channel is hollow (open at y = 40).
    """
    return [
        # Base with 2 upward notches
        ( 0,  0), (10,  0), (10,  6), (16,  6), (16,  0),
        (20,  0), (20,  6), (26,  6), (26,  0), (36,  0),
        # Right outer arm going up: 2 inward notches
        (36, 12), (30, 12), (30, 16), (36, 16),
        (36, 24), (30, 24), (30, 28), (36, 28),
        (36, 40),
        # Right inner corner, then inner walls of channel
        (28, 40), (28,  8), ( 8,  8), ( 8, 40),
        # Left inner corner
        ( 0, 40),
        # Left outer arm going down: 2 inward notches
        ( 0, 28), ( 6, 28), ( 6, 24), ( 0, 24),
        ( 0, 16), ( 6, 16), ( 6, 12), ( 0, 12),
    ]


# ---------------------------------------------------------------------------
# Placement helpers
# ---------------------------------------------------------------------------

def _bbox(pts):
    """Return (width, height) of the axis-aligned bounding box of pts."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(xs), max(ys)   # min is guaranteed to be at (0, 0)


def _auto_scale(pts, layout):
    """Return a float scale so the polygon fills one grid cell as fully as possible.

    Each cell has size  (CANVAS - 2*BORDER - (cols-1)*GAP) / cols  in width and
    the equivalent in height.  We take the smaller of the two axis factors so the
    scaled polygon still fits inside its cell.
    """
    w, h = _bbox(pts)
    cols, rows = layout
    max_cell_w = (CANVAS - 2 * BORDER - (cols - 1) * GAP) / cols
    max_cell_h = (CANVAS - 2 * BORDER - (rows - 1) * GAP) / rows
    return min(max_cell_w / w, max_cell_h / h)


def _scale_pts(pts, scale):
    """Return pts with every coordinate multiplied by *scale* and rounded."""
    return [(round(x * scale), round(y * scale)) for x, y in pts]


def _symmetric_positions(pts, layout):
    """Return (x_offset, y_offset) pairs for a centred symmetric grid.

    Parameters
    ----------
    pts    : polygon vertex list (used only to read bounding-box size)
    layout : (cols, rows) — grid dimensions

    The entire grid is centred on the CANVAS×CANVAS image.
    """
    w, h = _bbox(pts)
    cols, rows = layout

    grid_w = cols * w + (cols - 1) * GAP
    grid_h = rows * h + (rows - 1) * GAP

    start_x = (CANVAS - grid_w) // 2
    start_y = (CANVAS - grid_h) // 2

    positions = []
    for r in range(rows):
        for c in range(cols):
            x0 = start_x + c * (w + GAP)
            y0 = start_y + r * (h + GAP)
            positions.append((x0, y0))
    return positions


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_instance(draw: ImageDraw.ImageDraw, pts, x0, y0):
    """Render one polygon as a closed outline (no fill) onto *draw*."""
    world_pts = [(x0 + px, y0 + py) for px, py in pts]
    # Explicitly close the polygon for draw.line by appending the first point
    draw.line(world_pts + [world_pts[0]], fill=0, width=LINE_W)


def _generate(template_fn, layout):
    """Generate a single design image.

    Places copies of the template polygon in a symmetric grid and renders
    each as an outline-only polygon on a white canvas.
    Returns a (CANVAS, CANVAS) uint8 numpy array.
    """
    pts = template_fn()
    pts = _scale_pts(pts, _auto_scale(pts, layout))

    img  = Image.new('L', (CANVAS, CANVAS), color=255)
    draw = ImageDraw.Draw(img)

    for x0, y0 in _symmetric_positions(pts, layout):
        _draw_instance(draw, pts, x0, y0)

    return np.array(img)


# ---------------------------------------------------------------------------
# Design variant generators (public API — same signature as before)
# ---------------------------------------------------------------------------

def generate_design_A() -> np.ndarray:
    """Complex DIP outline, 4 copies in 2×2 symmetric grid."""
    return _generate(_template_A, (2, 2))


def generate_design_B() -> np.ndarray:
    """Winding staircase outline, 6 copies in 3×2 symmetric grid."""
    return _generate(_template_B, (3, 2))


def generate_design_C() -> np.ndarray:
    """Complex cross outline, 4 copies in 2×2 symmetric grid."""
    return _generate(_template_C, (2, 2))


def generate_design_D() -> np.ndarray:
    """Notched octagon outline, 6 copies in 2×3 symmetric grid."""
    return _generate(_template_D, (2, 3))


def generate_design_E() -> np.ndarray:
    """Ascending staircase band, 4 copies in 2×2 symmetric grid."""
    return _generate(_template_E, (2, 2))


def generate_design_F() -> np.ndarray:
    """T-shape with pin-slot notches, 4 copies in 2×2 symmetric grid."""
    return _generate(_template_F, (2, 2))


def generate_design_G() -> np.ndarray:
    """Hourglass with notched ends, 4 copies in 2×2 symmetric grid."""
    return _generate(_template_G, (2, 2))


def generate_design_H() -> np.ndarray:
    """Three-tooth comb, 4 copies in 2×2 symmetric grid."""
    return _generate(_template_H, (2, 2))


def generate_design_I() -> np.ndarray:
    """I-beam with notched flanges, 4 copies in 2×2 symmetric grid."""
    return _generate(_template_I, (2, 2))


def generate_design_J() -> np.ndarray:
    """U-channel with wall notches, 4 copies in 2×2 symmetric grid."""
    return _generate(_template_J, (2, 2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

GENERATORS = {
    'A': generate_design_A,
    'B': generate_design_B,
    'C': generate_design_C,
    'D': generate_design_D,
    'E': generate_design_E,
    'F': generate_design_F,
    'G': generate_design_G,
    'H': generate_design_H,
    'I': generate_design_I,
    'J': generate_design_J,
}


def generate_all(output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic PCB design images...")
    for name, gen_fn in GENERATORS.items():
        arr  = gen_fn()
        path = output_dir / f"design_{name}.png"
        Image.fromarray(arr.astype(np.uint8), mode='L').save(path)

        density = (arr < 128).mean() * 100
        print(f"  design_{name}.png  —  fill density: {density:.1f}%  →  {path}")

    print("Done.")


if __name__ == '__main__':
    generate_all()
