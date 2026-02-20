#!/usr/bin/env python3
"""Handcrafted feature extraction for PCB design images.

Features extracted (total = HAND_FEATURE_DIM = 22):

  Density & asymmetry (original 10):
    [0]    Global foreground (line) density
    [1-4]  Density in four quadrants (TL, TR, BL, BR)
    [5]    Left-Right asymmetry index  |left_density - right_density|
    [6]    Top-Bottom asymmetry index  |top_density - bottom_density|
    [7]    Horizontal-line ratio  (Sobel-Y response share)
    [8]    Vertical-line ratio    (Sobel-X response share)
    [9]    Diagonal-line ratio    (diagonal Sobel response share)

  Structural / layout (3):
    [10]   Aspect ratio (H / W)
    [11]   Connected component count  (log-normalised)
    [12]   Mean connected-component area (normalised by total image area)

  Symmetry (3):
    [13]   Left-Right symmetry  (Pearson correlation of left vs mirrored right)
    [14]   Top-Bottom symmetry  (Pearson correlation of top vs mirrored bottom)
    [15]   180-degree rotational symmetry

  Spatial distribution (3):
    [16]   Centre density  (foreground density in central 50 % area)
    [17]   Border density  (foreground density in outer border ring)
    [18]   Radial mean     (normalised mean distance of foreground from centre)

  Shape complexity (3):
    [19]   Perimeter ratio (edge pixels / foreground pixels)
    [20]   Hu moment 1  (eta_20 + eta_02, scale-invariant shape spread)
    [21]   Hu moment 2  ((eta_20 - eta_02)^2 + 4*eta_11^2, elongation)

Design images: white background (1.0), black lines (0.0), values in [0, 1].
Foreground pixels = pixels < 0.5.

Important: features are computed at the **original image resolution** (before any
resize) so that thin design lines are faithfully represented.  Downsampling with
LANCZOS interpolation blurs sub-pixel-wide lines into grey, which shifts the
binary threshold used for density and edge-ratio calculations.  Pass a PIL Image
(pre-resize) to preserve accuracy; a torch.Tensor path is kept for cases where
only a tensor is available (e.g. batch re-extraction).
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import label

HAND_FEATURE_DIM = 22

# Fixed Sobel kernels (computed once at module load)
_SOBEL_X = torch.tensor(
    [[-1., 0., 1.],
     [-2., 0., 2.],
     [-1., 0., 1.]], dtype=torch.float32
).view(1, 1, 3, 3)  # detects horizontal intensity gradient → responds to vertical lines

_SOBEL_Y = torch.tensor(
    [[-1., -2., -1.],
     [ 0.,  0.,  0.],
     [ 1.,  2.,  1.]], dtype=torch.float32
).view(1, 1, 3, 3)  # detects vertical intensity gradient → responds to horizontal lines

_SOBEL_D1 = torch.tensor(
    [[ 0.,  1.,  2.],
     [-1.,  0.,  1.],
     [-2., -1.,  0.]], dtype=torch.float32
).view(1, 1, 3, 3)  # 45-degree diagonal

_SOBEL_D2 = torch.tensor(
    [[-2., -1.,  0.],
     [-1.,  0.,  1.],
     [ 0.,  1.,  2.]], dtype=torch.float32
).view(1, 1, 3, 3)  # 135-degree diagonal


# ------------------------------------------------------------------
# Helper functions for new features
# ------------------------------------------------------------------

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two flattened arrays, clamped to [0, 1]."""
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    a_mean = a_flat.mean()
    b_mean = b_flat.mean()
    a_std = a_flat.std()
    b_std = b_flat.std()
    if a_std < 1e-8 or b_std < 1e-8:
        # If either half is near-constant, symmetry is perfect if both are constant
        return 1.0 if (a_std < 1e-8 and b_std < 1e-8) else 0.0
    cov = ((a_flat - a_mean) * (b_flat - b_mean)).mean()
    r = cov / (a_std * b_std)
    return float(np.clip(r, 0.0, 1.0))


def _symmetry_features(fg: np.ndarray) -> tuple[float, float, float]:
    """Compute LR, TB, and 180-degree rotational symmetry scores."""
    H, W = fg.shape
    h2, w2 = H // 2, W // 2

    # Left-Right: compare left half with horizontally flipped right half
    left = fg[:, :w2]
    right_flipped = fg[:, -1 : W - w2 - 1 : -1] if w2 > 0 else fg[:, :w2]
    right_flipped = np.flip(fg[:, w2 : 2 * w2], axis=1)
    lr_sym = _pearson(left, right_flipped)

    # Top-Bottom: compare top half with vertically flipped bottom half
    top = fg[:h2, :]
    bot_flipped = np.flip(fg[h2 : 2 * h2, :], axis=0)
    tb_sym = _pearson(top, bot_flipped)

    # 180-degree rotation
    rot180 = np.flip(fg, axis=(0, 1))
    rot_sym = _pearson(fg, rot180)

    return lr_sym, tb_sym, rot_sym


def _component_features(fg_np: np.ndarray) -> tuple[float, float]:
    """Connected component count (log-normalised) and mean component area."""
    labeled, num_components = label(fg_np)
    log_count = float(np.log1p(num_components))  # log(1 + n) for stability
    total_area = float(fg_np.shape[0] * fg_np.shape[1])
    if num_components > 0:
        component_sizes = np.bincount(labeled.ravel())[1:]  # skip background label 0
        mean_area = float(component_sizes.mean()) / total_area
    else:
        mean_area = 0.0
    return log_count, mean_area


def _spatial_distribution(fg: np.ndarray) -> tuple[float, float, float]:
    """Centre density, border density, and radial mean distance."""
    H, W = fg.shape
    h4, w4 = H // 4, W // 4

    # Centre: inner 50% area (from 25% to 75% on each axis)
    centre_region = fg[h4 : H - h4, w4 : W - w4]
    centre_density = float(centre_region.mean())

    # Border: everything outside the centre region
    border_mask = np.ones_like(fg, dtype=bool)
    border_mask[h4 : H - h4, w4 : W - w4] = False
    border_density = float(fg[border_mask].mean()) if border_mask.any() else 0.0

    # Radial mean: average normalised distance of foreground pixels from centre
    cy, cx = H / 2.0, W / 2.0
    max_r = np.sqrt(cy ** 2 + cx ** 2)
    ys, xs = np.where(fg > 0.5)
    if len(ys) > 0:
        dists = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        radial_mean = float(dists.mean() / max_r)
    else:
        radial_mean = 0.0

    return centre_density, border_density, radial_mean


def _perimeter_ratio(fg: np.ndarray) -> float:
    """Edge pixels / foreground pixels — measures outline complexity."""
    fg_bool = fg > 0.5
    # Erode by 1 pixel: foreground pixels that have at least one background neighbour
    from scipy.ndimage import binary_erosion
    interior = binary_erosion(fg_bool)
    edge_pixels = fg_bool.astype(np.int32) - interior.astype(np.int32)
    n_fg = int(fg_bool.sum())
    n_edge = int(edge_pixels.sum())
    if n_fg == 0:
        return 0.0
    return float(n_edge) / float(n_fg)


def _hu_moments(fg: np.ndarray) -> tuple[float, float]:
    """First two Hu moments from the binary foreground mask.

    Hu1 = eta_20 + eta_02            (overall shape spread)
    Hu2 = (eta_20 - eta_02)^2 + 4*eta_11^2  (elongation / directionality)

    Returned in log scale: sign(h) * log10(|h| + 1e-12) for numerical stability.
    """
    ys, xs = np.where(fg > 0.5)
    n = len(ys)
    if n == 0:
        return 0.0, 0.0

    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)

    # Centroid
    cx = xs.mean()
    cy = ys.mean()

    # Central moments
    dx = xs - cx
    dy = ys - cy
    mu00 = float(n)
    mu20 = float((dx ** 2).sum())
    mu02 = float((dy ** 2).sum())
    mu11 = float((dx * dy).sum())

    # Normalised central moments
    norm = mu00 ** 2.0  # (p+q)/2 + 1 = 2 for second-order moments
    eta20 = mu20 / norm
    eta02 = mu02 / norm
    eta11 = mu11 / norm

    hu1 = eta20 + eta02
    hu2 = (eta20 - eta02) ** 2 + 4.0 * eta11 ** 2

    # Log scale for numerical stability
    def _log_hu(h: float) -> float:
        return float(np.sign(h) * np.log10(abs(h) + 1e-12))

    return _log_hu(hu1), _log_hu(hu2)


# ------------------------------------------------------------------
# Main extraction function
# ------------------------------------------------------------------

def extract_handcrafted_features(design_input) -> torch.Tensor:
    """Compute handcrafted features from a design image at its native resolution.

    Args:
        design_input: PIL.Image.Image  — preferred; features are computed at the
                          original image size before any resize, so thin lines are
                          not blurred by interpolation.
                      torch.Tensor of shape (1, H, W) or (H, W)  — accepted when
                          only a tensor is available (e.g. batch re-extraction).
                          Values must be in [0, 1] (white=1.0, black lines=0.0).

    Returns:
        features: Tensor of shape (HAND_FEATURE_DIM,) = (22,), dtype=float32.
    """
    if isinstance(design_input, Image.Image):
        arr = np.array(design_input.convert('L'), dtype=np.float32) / 255.0
        img = torch.from_numpy(arr)          # (H, W) at original resolution
    elif isinstance(design_input, torch.Tensor):
        img = design_input.squeeze(0) if design_input.dim() == 3 else design_input
    else:
        raise TypeError(
            f"design_input must be a PIL Image or torch.Tensor, got {type(design_input)}"
        )

    H, W = img.shape
    h2, w2 = H // 2, W // 2

    # Binary foreground mask: black lines = 1, white background = 0
    fg = (img < 0.5).float()
    fg_np = fg.numpy()

    # === Original features [0-9] ===

    # --- Spatial density features ---
    global_density = fg.mean().item()
    q_tl = fg[:h2, :w2].mean().item()   # top-left
    q_tr = fg[:h2, w2:].mean().item()   # top-right
    q_bl = fg[h2:, :w2].mean().item()   # bottom-left
    q_br = fg[h2:, w2:].mean().item()   # bottom-right

    left_density  = fg[:, :w2].mean().item()
    right_density = fg[:, w2:].mean().item()
    top_density   = fg[:h2, :].mean().item()
    bot_density   = fg[h2:, :].mean().item()

    lr_asym = abs(left_density  - right_density)
    tb_asym = abs(top_density   - bot_density)

    # --- Edge orientation features (Sobel on raw intensity) ---
    img_4d = img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    device = img.device

    kernels = [_SOBEL_X, _SOBEL_Y, _SOBEL_D1, _SOBEL_D2]
    kernels = [k.to(device) for k in kernels]

    with torch.no_grad():
        gx  = F.conv2d(img_4d, kernels[0], padding=1).abs().mean().item()
        gy  = F.conv2d(img_4d, kernels[1], padding=1).abs().mean().item()
        gd1 = F.conv2d(img_4d, kernels[2], padding=1).abs().mean().item()
        gd2 = F.conv2d(img_4d, kernels[3], padding=1).abs().mean().item()

    gd_avg = (gd1 + gd2) / 2.0
    g_total = gx + gy + gd_avg + 1e-8

    h_ratio = gy  / g_total   # horizontal-line content (gy responds to horizontal lines)
    v_ratio = gx  / g_total   # vertical-line content
    d_ratio = gd_avg / g_total

    # === New features [10-21] ===

    # [10] Aspect ratio
    aspect_ratio = float(H) / float(W) if W > 0 else 1.0

    # [11-12] Connected component features
    log_num_components, mean_component_area = _component_features(fg_np)

    # [13-15] Symmetry features
    lr_sym, tb_sym, rot180_sym = _symmetry_features(fg_np)

    # [16-18] Spatial distribution features
    centre_density, border_density, radial_mean = _spatial_distribution(fg_np)

    # [19] Perimeter ratio (edge complexity)
    peri_ratio = _perimeter_ratio(fg_np)

    # [20-21] Hu moments
    hu1, hu2 = _hu_moments(fg_np)

    features = torch.tensor(
        [global_density, q_tl, q_tr, q_bl, q_br,
         lr_asym, tb_asym, h_ratio, v_ratio, d_ratio,
         aspect_ratio, log_num_components, mean_component_area,
         lr_sym, tb_sym, rot180_sym,
         centre_density, border_density, radial_mean,
         peri_ratio, hu1, hu2],
        dtype=torch.float32
    )
    return features


def batch_extract_handcrafted_features(design_batch: torch.Tensor) -> torch.Tensor:
    """Compute handcrafted features for a batch of design images.

    Args:
        design_batch: Tensor of shape (B, 1, H, W).

    Returns:
        Tensor of shape (B, HAND_FEATURE_DIM).
    """
    return torch.stack(
        [extract_handcrafted_features(design_batch[i]) for i in range(design_batch.size(0))]
    )
