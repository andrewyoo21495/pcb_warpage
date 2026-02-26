"""Core preprocessing functions: downsampling, outlier detection, interpolation, smoothing."""

import logging
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


def downsample_median(data: np.ndarray, factor: int) -> np.ndarray:
    """Downsample data by computing the median of non-NaN values in each block.

    Args:
        data: Input array of shape (H, W), may contain NaN.
        factor: Downsampling factor (e.g., 4 means 1/4 resolution).

    Returns:
        Downsampled array of shape (H // factor, W // factor).
    """
    H, W = data.shape
    new_H = H // factor
    new_W = W // factor

    # Truncate to exact multiple of factor
    trimmed = data[:new_H * factor, :new_W * factor]

    # Vectorized: reshape into blocks and compute nanmedian in one call
    reshaped = trimmed.reshape(new_H, factor, new_W, factor)
    reshaped = reshaped.transpose(0, 2, 1, 3).reshape(new_H, new_W, factor * factor)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmedian(reshaped, axis=2)

    return result


def detect_and_remove_outliers(
    data: np.ndarray,
    grid_size: int = 8,
    z_threshold: float = 3.0,
) -> tuple:
    """Detect outliers per region using z-score and replace with NaN.

    Args:
        data: Input array of shape (H, W), may contain NaN.
        grid_size: Number of grid divisions along each axis.
        z_threshold: Z-score threshold for outlier detection.

    Returns:
        (result_array, total_outliers_removed)
    """
    result = data.copy()
    H, W = result.shape
    total_removed = 0

    row_edges = np.linspace(0, H, grid_size + 1, dtype=int)
    col_edges = np.linspace(0, W, grid_size + 1, dtype=int)

    for ri in range(grid_size):
        for ci in range(grid_size):
            r_start, r_end = row_edges[ri], row_edges[ri + 1]
            c_start, c_end = col_edges[ci], col_edges[ci + 1]
            region = result[r_start:r_end, c_start:c_end]

            valid_mask = ~np.isnan(region)
            valid_vals = region[valid_mask]

            if len(valid_vals) < 2:
                continue

            mean = np.mean(valid_vals)
            std = np.std(valid_vals, ddof=0)

            if std == 0:
                continue

            z_scores = np.abs((region - mean) / std)
            outlier_mask = valid_mask & (z_scores > z_threshold)
            n_outliers = np.sum(outlier_mask)

            if n_outliers > 0:
                region[outlier_mask] = np.nan
                total_removed += n_outliers

    return result, int(total_removed)


def interpolate_surface(
    data: np.ndarray,
    poly_degree: int = 3,
    ridge_alpha: float = 0.1,
) -> tuple:
    """Fill NaN values using polynomial surface regression with ridge regularization.

    Args:
        data: Input array of shape (H, W), may contain NaN.
        poly_degree: Degree of polynomial features.
        ridge_alpha: Ridge regularization coefficient.

    Returns:
        (result_array, n_interpolated) — array with NaNs filled, and count of filled pixels.
    """
    H, W = data.shape
    total_pixels = H * W

    rows, cols = np.indices((H, W))
    rows_flat = rows.ravel().astype(np.float64)
    cols_flat = cols.ravel().astype(np.float64)
    vals_flat = data.ravel()

    valid_mask = ~np.isnan(vals_flat)
    nan_mask = np.isnan(vals_flat)

    n_valid = np.sum(valid_mask)
    n_interpolated = int(np.sum(nan_mask))
    valid_ratio = n_valid / total_pixels

    if n_valid == 0:
        logger.warning("No valid values — cannot interpolate.")
        return data, 0

    if valid_ratio < 0.05:
        logger.warning("Valid values: %.1f%% (< 5%%) — potential quality degradation.",
                        valid_ratio * 100)
    elif valid_ratio < 0.10:
        logger.warning("Valid values: %.1f%% (< 10%%) — quality may be degraded.",
                        valid_ratio * 100)

    # Normalize coordinates for numerical stability
    row_norm = rows_flat / max(H - 1, 1)
    col_norm = cols_flat / max(W - 1, 1)
    coords = np.column_stack([row_norm, col_norm])

    # Polynomial features
    poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
    X_all = poly.fit_transform(coords)

    X_train = X_all[valid_mask]
    y_train = vals_flat[valid_mask]

    # Fit ridge regression
    model = Ridge(alpha=ridge_alpha)
    model.fit(X_train, y_train)

    # Predict NaN positions only
    result = data.copy()
    if np.any(nan_mask):
        X_predict = X_all[nan_mask]
        predicted = model.predict(X_predict)
        result_flat = result.ravel()
        result_flat[nan_mask] = predicted
        result = result_flat.reshape(H, W)

    return result, n_interpolated


def smooth_gaussian(
    data: np.ndarray,
    sigma: float = 2.0,
    iterations: int = 3,
) -> np.ndarray:
    """Apply iterative Gaussian smoothing while preserving the original min/max.

    Sigma is adaptive: scaled proportionally to the data dimensions so that
    the visual smoothness is consistent regardless of resolution.  Different
    sigma values are used for rows and columns (anisotropic) to handle
    non-square data correctly.

    Each iteration applies a Gaussian filter and then linearly rescales the
    result so that its min and max match the original data.  Repeating this
    process produces a surface where transitions from peaks to valleys are
    very smooth and natural, while the extreme values are preserved.

    Args:
        data: Input array (must be NaN-free).
        sigma: Base smoothing factor. The actual kernel sigma for each axis
               is ``max(1.0, axis_length * sigma / 100)``.
        iterations: Number of smooth-then-rescale iterations.

    Returns:
        Smoothed array of the same shape with original min/max preserved.
    """
    rows, cols = data.shape
    sigma_row = max(1.0, rows * sigma / 100)
    sigma_col = max(1.0, cols * sigma / 100)

    orig_min = np.min(data)
    orig_max = np.max(data)

    if orig_max - orig_min < 1e-12:
        return data.copy()

    smoothed = data.copy()
    for _ in range(iterations):
        smoothed = gaussian_filter(smoothed, sigma=[sigma_row, sigma_col])

        s_min = np.min(smoothed)
        s_max = np.max(smoothed)

        if s_max - s_min < 1e-12:
            break

        smoothed = (smoothed - s_min) / (s_max - s_min)
        smoothed = smoothed * (orig_max - orig_min) + orig_min

    return smoothed
