#!/usr/bin/env python3
"""Handcrafted feature extraction for PCB design images.

Features extracted (total = HAND_FEATURE_DIM = 10):
  [0]    Global foreground (line) density
  [1-4]  Density in four quadrants (TL, TR, BL, BR)
  [5]    Left-Right asymmetry index  |left_density - right_density|
  [6]    Top-Bottom asymmetry index  |top_density - bottom_density|
  [7]    Horizontal-line ratio  (Sobel-Y response share)
  [8]    Vertical-line ratio    (Sobel-X response share)
  [9]    Diagonal-line ratio    (diagonal Sobel response share)

Design images: white background (1.0), black lines (0.0), values in [0, 1].
Foreground pixels = pixels < 0.5.
"""

import torch
import torch.nn.functional as F

HAND_FEATURE_DIM = 10

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


def extract_handcrafted_features(design_tensor: torch.Tensor) -> torch.Tensor:
    """Compute handcrafted features from a single design image tensor.

    Args:
        design_tensor: Tensor of shape (1, H, W) or (H, W),
                       values in [0, 1] (white=1.0, black lines=0.0).

    Returns:
        features: Tensor of shape (HAND_FEATURE_DIM,) = (10,), dtype=float32.
    """
    if design_tensor.dim() == 3:
        img = design_tensor.squeeze(0)  # (H, W)
    else:
        img = design_tensor  # (H, W)

    H, W = img.shape
    h2, w2 = H // 2, W // 2

    # Binary foreground mask: black lines = 1, white background = 0
    fg = (img < 0.5).float()

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

    features = torch.tensor(
        [global_density, q_tl, q_tr, q_bl, q_br,
         lr_asym, tb_asym, h_ratio, v_ratio, d_ratio],
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
