"""
PCB Warpage Feature Documentation Generator
=============================================
Generates a standalone HTML document that explains warpage feature variables
with synthetic surface visualizations embedded as base64 images.

Usage:
    python preprocess/generate_feature_doc.py

Output:
    preprocess/warpage_features_doc.html
"""

import base64
import io
import os
import textwrap

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter, label
from scipy.stats import skew, kurtosis


# =============================================================================
# Utility: figure → base64 PNG string
# =============================================================================

def fig_to_base64(fig, dpi=130):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# =============================================================================
# Synthetic warpage surface generators
# =============================================================================

def make_grid(size=128):
    """Create normalized coordinate grids in [-1, 1]."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    return X, Y


def surface_crying(size=128, amp=500):
    """Crying mode: center is higher than edges."""
    X, Y = make_grid(size)
    R2 = X**2 + Y**2
    Z = amp * (1 - R2)
    Z -= Z.min()
    return Z


def surface_smiling(size=128, amp=500):
    """Smiling mode: edges are higher than center."""
    X, Y = make_grid(size)
    R2 = X**2 + Y**2
    Z = amp * R2
    Z -= Z.min()
    return Z


def surface_saddle(size=128, amp=400):
    """Saddle mode: opposite corners high/low."""
    X, Y = make_grid(size)
    Z = amp * (X**2 - Y**2)
    Z -= Z.min()
    return Z


def surface_twist(size=128, amp=300):
    """Twist mode: diagonal warpage."""
    X, Y = make_grid(size)
    Z = amp * X * Y
    Z -= Z.min()
    return Z


def surface_complex(size=128):
    """A realistic-looking complex warpage surface."""
    X, Y = make_grid(size)
    Z = (300 * (1 - X**2 - Y**2)
         + 150 * np.sin(2 * np.pi * X) * np.cos(np.pi * Y)
         + 80 * np.exp(-((X - 0.3)**2 + (Y + 0.2)**2) / 0.1))
    Z -= Z.min()
    # Add slight noise
    rng = np.random.RandomState(42)
    Z += rng.normal(0, 5, Z.shape)
    Z = gaussian_filter(Z, sigma=2)
    Z -= Z.min()
    return Z


def surface_with_mask(size=128):
    """Create a surface with NaN holes simulating component masking."""
    Z = surface_complex(size)
    mask = np.zeros_like(Z, dtype=bool)
    # Simulate rectangular component masks
    rng = np.random.RandomState(7)
    for _ in range(8):
        r, c = rng.randint(10, size - 30), rng.randint(10, size - 30)
        h, w = rng.randint(8, 20), rng.randint(8, 20)
        mask[r:r+h, c:c+w] = True
    Z_masked = Z.copy()
    Z_masked[mask] = np.nan
    return Z, Z_masked, mask


# =============================================================================
# Image generators for each section
# =============================================================================

def gen_overview_image():
    """Overview: show a sample warpage heatmap with 3D view."""
    Z = surface_complex()
    fig = plt.figure(figsize=(12, 4.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])

    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(Z, cmap='jet', origin='lower')
    ax1.set_title('PCB Warpage Heatmap (2D)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('X (pixel)')
    ax1.set_ylabel('Y (pixel)')
    plt.colorbar(im, ax=ax1, label='Elevation (μm)', shrink=0.85)

    ax2 = fig.add_subplot(gs[1], projection='3d')
    X, Y = make_grid(Z.shape[0])
    ax2.plot_surface(X, Y, Z, cmap='jet', alpha=0.9, linewidth=0,
                     antialiased=True, rcount=80, ccount=80)
    ax2.set_title('3D Surface View', fontsize=13, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Elevation (μm)')
    ax2.view_init(elev=30, azim=-60)

    fig.tight_layout()
    return fig_to_base64(fig)


def gen_basic_stats_image():
    """Basic statistics: histogram + surface with annotations."""
    Z = surface_complex()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Histogram
    ax = axes[0]
    vals = Z.ravel()
    ax.hist(vals, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
    mean_v = np.mean(vals)
    std_v = np.std(vals)
    max_v = np.max(vals)
    p95 = np.percentile(vals, 95)
    ax.axvline(mean_v, color='red', ls='--', lw=2, label=f'Mean = {mean_v:.1f}')
    ax.axvline(max_v, color='darkred', ls='-', lw=2, label=f'Max = {max_v:.1f}')
    ax.axvline(p95, color='orange', ls='--', lw=2, label=f'P95 = {p95:.1f}')
    ax.axvspan(mean_v - std_v, mean_v + std_v, alpha=0.15, color='red',
               label=f'±1σ (σ={std_v:.1f})')
    ax.set_title('Warpage Value Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Elevation (μm)')
    ax.set_ylabel('Pixel Count')
    ax.legend(fontsize=9)

    # Skewness & Kurtosis illustration
    sk = skew(vals)
    ku = kurtosis(vals)
    ax.text(0.98, 0.75, f'Skewness = {sk:.3f}\nKurtosis = {ku:.3f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

    # Surface with max location
    ax2 = axes[1]
    im = ax2.imshow(Z, cmap='jet', origin='lower')
    max_idx = np.unravel_index(np.argmax(Z), Z.shape)
    ax2.plot(max_idx[1], max_idx[0], 'r*', markersize=18, markeredgecolor='white',
             markeredgewidth=1.5, label='Max location')
    # Mean contour
    ax2.contour(Z, levels=[mean_v], colors='white', linewidths=1.5, linestyles='--')
    ax2.set_title('Surface with Max Location & Mean Contour', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    plt.colorbar(im, ax=ax2, label='Elevation (μm)', shrink=0.85)

    fig.tight_layout()
    return fig_to_base64(fig)


def gen_warpage_mode_image():
    """Warpage modes: Crying, Smiling, Saddle, Twist."""
    surfaces = [
        (surface_crying(), 'Crying (Convex)',
         'Center > Edge\ncenter_vs_edge > 1'),
        (surface_smiling(), 'Smiling (Concave)',
         'Edge > Center\ncenter_vs_edge < 1'),
        (surface_saddle(), 'Saddle',
         'Opposite edges differ\nsaddle_index ≠ 0'),
        (surface_twist(), 'Twist',
         'Diagonal warpage\ncorner_std high'),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7),
                              gridspec_kw={'height_ratios': [3, 1]})

    for i, (Z, title, desc) in enumerate(surfaces):
        ax = axes[0, i]
        im = ax.imshow(Z, cmap='jet', origin='lower')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.04)

        # Cross-section through center
        ax2 = axes[1, i]
        mid = Z.shape[0] // 2
        ax2.plot(Z[mid, :], 'b-', lw=2, label='Row center')
        ax2.plot(Z[:, mid], 'r--', lw=2, label='Col center')
        ax2.set_xlabel('Position', fontsize=9)
        ax2.set_ylabel('μm', fontsize=9)
        ax2.legend(fontsize=7, loc='best')
        ax2.set_title(desc, fontsize=9, color='gray')
        ax2.tick_params(labelsize=8)

    fig.suptitle('Warpage Mode Classification', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig_to_base64(fig)


def gen_spatial_distribution_image():
    """Spatial distribution: centroid, quadrants, max location."""
    Z = surface_complex()
    H, W = Z.shape

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1) Weighted centroid
    ax = axes[0]
    im = ax.imshow(Z, cmap='jet', origin='lower')
    rows, cols = np.indices(Z.shape)
    total = Z.sum()
    cy = (rows * Z).sum() / total
    cx = (cols * Z).sum() / total
    ax.plot(cx, cy, 'w+', markersize=20, markeredgewidth=3, label='Centroid')
    ax.plot(W/2, H/2, 'kx', markersize=15, markeredgewidth=2, label='Geometric center')
    ax.set_title('Warpage Centroid', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    plt.colorbar(im, ax=ax, shrink=0.85)

    # 2) Quadrant analysis
    ax = axes[1]
    q_means = np.zeros((2, 2))
    q_means[0, 0] = Z[:H//2, :W//2].mean()
    q_means[0, 1] = Z[:H//2, W//2:].mean()
    q_means[1, 0] = Z[H//2:, :W//2].mean()
    q_means[1, 1] = Z[H//2:, W//2:].mean()
    im2 = ax.imshow(Z, cmap='jet', origin='lower', alpha=0.5)
    # Draw quadrant boundaries
    ax.axhline(H//2, color='white', lw=2)
    ax.axvline(W//2, color='white', lw=2)
    # Label quadrant means
    for r in range(2):
        for c in range(2):
            y_pos = H//4 + r * H//2
            x_pos = W//4 + c * W//2
            ax.text(x_pos, y_pos, f'Q{r*2+c+1}\n{q_means[r,c]:.0f}μm',
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round', fc='black', alpha=0.6))
    ax.set_title('Quadrant Mean Warpage', fontsize=12, fontweight='bold')
    q_diff = q_means.max() - q_means.min()
    ax.text(0.5, -0.12, f'Quadrant max diff = {q_diff:.0f} μm',
            transform=ax.transAxes, ha='center', fontsize=10, color='gray')

    # 3) High warpage area
    ax = axes[2]
    threshold = np.percentile(Z, 90)
    high_mask = Z >= threshold
    ax.imshow(Z, cmap='gray', origin='lower', alpha=0.4)
    ax.imshow(np.where(high_mask, 1, np.nan), cmap='Reds', origin='lower',
              alpha=0.7, vmin=0, vmax=1)
    ratio = high_mask.sum() / Z.size * 100
    ax.set_title(f'High Warpage Area (≥P90)', fontsize=12, fontweight='bold')
    ax.text(0.5, -0.12, f'Area ratio = {ratio:.1f}%  (threshold = {threshold:.0f} μm)',
            transform=ax.transAxes, ha='center', fontsize=10, color='gray')

    fig.tight_layout()
    return fig_to_base64(fig)


def gen_surface_geometry_image():
    """Surface geometry: gradient magnitude, curvature, roughness."""
    Z = surface_complex()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Original
    ax = axes[0]
    im = ax.imshow(Z, cmap='jet', origin='lower')
    ax.set_title('Original Surface', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.85)

    # Gradient magnitude
    gy, gx = np.gradient(Z)
    grad_mag = np.sqrt(gx**2 + gy**2)
    ax = axes[1]
    im = ax.imshow(grad_mag, cmap='hot', origin='lower')
    ax.set_title(f'Gradient Magnitude\nmean={grad_mag.mean():.2f}, max={grad_mag.max():.2f}',
                 fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.85)

    # Laplacian (mean curvature proxy)
    lap = np.gradient(np.gradient(Z, axis=0), axis=0) + np.gradient(np.gradient(Z, axis=1), axis=1)
    ax = axes[2]
    vlim = np.percentile(np.abs(lap), 98)
    im = ax.imshow(lap, cmap='RdBu_r', origin='lower', vmin=-vlim, vmax=vlim)
    ax.set_title(f'Laplacian (Curvature)\nmean={lap.mean():.2f}',
                 fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.85)

    # Surface roughness
    smooth = gaussian_filter(Z, sigma=5)
    roughness = Z - smooth
    ax = axes[3]
    vlim_r = np.percentile(np.abs(roughness), 98)
    im = ax.imshow(roughness, cmap='coolwarm', origin='lower', vmin=-vlim_r, vmax=vlim_r)
    Ra = np.mean(np.abs(roughness))
    Rq = np.sqrt(np.mean(roughness**2))
    ax.set_title(f'Surface Roughness\nRa={Ra:.2f}, Rq={Rq:.2f}',
                 fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.85)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig_to_base64(fig)


def gen_symmetry_image():
    """Symmetry: LR, TB, rotational comparisons."""
    Z = surface_complex()
    H, W = Z.shape

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    titles_top = ['Left-Right Flip', 'Top-Bottom Flip', '180° Rotation']
    transforms = [
        np.fliplr(Z),
        np.flipud(Z),
        np.rot90(Z, 2),
    ]

    for i, (Zt, title) in enumerate(zip(transforms, titles_top)):
        # Difference map
        diff = Z - Zt
        sym_score = np.mean(np.abs(diff)) / (Z.max() - Z.min())

        ax = axes[0, i]
        ax.imshow(Z, cmap='jet', origin='lower', alpha=0.5)
        ax.set_title(f'{title}', fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.axvline(W//2, color='white', lw=2, ls='--')
            ax.set_ylabel('Original', fontsize=10)
        elif i == 1:
            ax.axhline(H//2, color='white', lw=2, ls='--')

        ax2 = axes[1, i]
        vlim = np.percentile(np.abs(diff), 98)
        im = ax2.imshow(diff, cmap='RdBu_r', origin='lower', vmin=-vlim, vmax=vlim)
        ax2.set_title(f'Difference Map\nAsymmetry = {sym_score:.4f}',
                      fontsize=11, fontweight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im, ax=ax2, shrink=0.75)
        if i == 0:
            ax2.set_ylabel('|Original - Transformed|', fontsize=10)

    fig.suptitle('Symmetry Analysis (lower asymmetry = more symmetric)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig_to_base64(fig)


def gen_frequency_image():
    """Frequency domain: FFT power spectrum, low/high frequency ratio."""
    Z = surface_complex()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Original
    ax = axes[0]
    im = ax.imshow(Z, cmap='jet', origin='lower')
    ax.set_title('Original Surface', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.85)

    # 2D FFT power spectrum
    fft2 = np.fft.fft2(Z)
    fft_shift = np.fft.fftshift(fft2)
    power = np.log1p(np.abs(fft_shift)**2)

    ax = axes[1]
    im = ax.imshow(power, cmap='inferno', origin='lower')
    ax.set_title('2D Power Spectrum (log)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Freq X')
    ax.set_ylabel('Freq Y')
    # Mark center
    cy, cx = power.shape[0]//2, power.shape[1]//2
    circle = plt.Circle((cx, cy), radius=power.shape[0]//8, fill=False,
                         color='cyan', lw=2, ls='--', label='Low-freq region')
    ax.add_patch(circle)
    ax.legend(fontsize=8, loc='lower right')
    plt.colorbar(im, ax=ax, shrink=0.85)

    # Low vs High frequency decomposition
    ax = axes[2]
    # Low-pass filter
    mask_lf = np.zeros_like(Z)
    r = Z.shape[0] // 8
    mask_freq = np.zeros_like(power)
    Y_idx, X_idx = np.ogrid[:Z.shape[0], :Z.shape[1]]
    dist = np.sqrt((Y_idx - cy)**2 + (X_idx - cx)**2)
    low_mask = dist <= r

    total_energy = np.sum(np.abs(fft_shift)**2)
    low_energy = np.sum(np.abs(fft_shift[low_mask])**2)
    ratio = low_energy / total_energy * 100

    # Reconstruct low-freq surface
    fft_low = fft_shift.copy()
    fft_low[~low_mask] = 0
    Z_low = np.real(np.fft.ifft2(np.fft.ifftshift(fft_low)))

    Z_high = Z - Z_low

    ax.plot(Z[Z.shape[0]//2, :], 'b-', lw=2, label='Original', alpha=0.7)
    ax.plot(Z_low[Z.shape[0]//2, :], 'r-', lw=2, label=f'Low-freq ({ratio:.1f}%)')
    ax.plot(Z_high[Z.shape[0]//2, :], 'g--', lw=1.5, label=f'High-freq ({100-ratio:.1f}%)')
    ax.set_title('Frequency Decomposition\n(center row cross-section)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Elevation (μm)')
    ax.legend(fontsize=9)

    for a in axes[:2]:
        a.set_xticks([])
        a.set_yticks([])

    fig.tight_layout()
    return fig_to_base64(fig)


def gen_masking_image():
    """Masking: NaN region analysis before interpolation."""
    Z_full, Z_masked, mask = surface_with_mask()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Full surface
    ax = axes[0]
    im = ax.imshow(Z_full, cmap='jet', origin='lower')
    ax.set_title('Complete Surface\n(ground truth)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.85)

    # Masked surface
    ax = axes[1]
    im = ax.imshow(Z_masked, cmap='jet', origin='lower')
    ax.set_title('Masked Surface\n(with component holes)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.85)

    # Mask analysis
    ax = axes[2]
    labeled, n_components = label(mask)
    ax.imshow(labeled, cmap='tab20', origin='lower', interpolation='nearest')
    missing_ratio = mask.sum() / mask.size * 100
    # Find largest component
    comp_sizes = []
    for i in range(1, n_components + 1):
        comp_sizes.append((labeled == i).sum())
    largest = max(comp_sizes) if comp_sizes else 0
    ax.set_title(f'Connected Components\n{n_components} regions, missing={missing_ratio:.1f}%',
                 fontsize=11, fontweight='bold')
    ax.text(0.5, -0.12, f'Largest component = {largest} pixels',
            transform=ax.transAxes, ha='center', fontsize=10, color='gray')

    for a in axes:
        a.set_xticks([])
        a.set_yticks([])

    fig.tight_layout()
    return fig_to_base64(fig)


def gen_crosssection_image():
    """Cross-section profiles: row, column, diagonal."""
    Z = surface_complex()
    H, W = Z.shape

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3])

    # Surface with cross-section lines
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(Z, cmap='jet', origin='lower')
    mid_r, mid_c = H // 2, W // 2
    # Row line
    ax1.axhline(mid_r, color='cyan', lw=2, ls='-', label='Center row')
    # Col line
    ax1.axvline(mid_c, color='yellow', lw=2, ls='-', label='Center col')
    # Diagonal
    ax1.plot([0, W-1], [0, H-1], 'w--', lw=2, label='Diagonal')
    ax1.set_title('Cross-section Lines', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Profiles
    ax2 = fig.add_subplot(gs[1])
    row_profile = Z[mid_r, :]
    col_profile = Z[:, mid_c]
    diag_idx = np.linspace(0, min(H, W) - 1, min(H, W)).astype(int)
    diag_profile = Z[diag_idx, diag_idx]

    x_row = np.linspace(0, 1, len(row_profile))
    x_col = np.linspace(0, 1, len(col_profile))
    x_diag = np.linspace(0, 1, len(diag_profile))

    ax2.plot(x_row, row_profile, 'c-', lw=2,
             label=f'Center row (range={row_profile.max()-row_profile.min():.0f}μm)')
    ax2.plot(x_col, col_profile, 'y-', lw=2,
             label=f'Center col (range={col_profile.max()-col_profile.min():.0f}μm)')
    ax2.plot(x_diag, diag_profile, 'w-', lw=2,
             label=f'Diagonal (range={diag_profile.max()-diag_profile.min():.0f}μm)')
    ax2.set_facecolor('#2a2a2a')
    ax2.set_title('Cross-section Profiles', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Normalized Position (0→1)')
    ax2.set_ylabel('Elevation (μm)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig_to_base64(fig)


# =============================================================================
# HTML Template
# =============================================================================

CSS_STYLE = """
:root {
    --primary: #1a73e8;
    --primary-light: #e8f0fe;
    --bg: #f8f9fa;
    --card-bg: #ffffff;
    --text: #202124;
    --text-secondary: #5f6368;
    --border: #dadce0;
    --accent-green: #1e8e3e;
    --accent-orange: #e37400;
    --accent-red: #d93025;
    --code-bg: #f1f3f4;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.7;
}
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px 30px;
}
.header {
    background: linear-gradient(135deg, #1a73e8, #174ea6);
    color: white;
    padding: 50px 40px;
    border-radius: 16px;
    margin-bottom: 40px;
    box-shadow: 0 4px 20px rgba(26,115,232,0.3);
}
.header h1 { font-size: 2.2em; margin-bottom: 10px; }
.header p { font-size: 1.1em; opacity: 0.9; }
.toc {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 30px 35px;
    margin-bottom: 35px;
    border: 1px solid var(--border);
}
.toc h2 { font-size: 1.3em; margin-bottom: 15px; color: var(--primary); }
.toc ol { padding-left: 20px; }
.toc li { margin: 8px 0; }
.toc a { color: var(--primary); text-decoration: none; font-size: 1.05em; }
.toc a:hover { text-decoration: underline; }
.section {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 35px 40px;
    margin-bottom: 30px;
    border: 1px solid var(--border);
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.section h2 {
    font-size: 1.6em;
    color: var(--primary);
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--primary-light);
}
.section h3 {
    font-size: 1.2em;
    color: var(--text);
    margin: 25px 0 12px 0;
}
.section p {
    margin-bottom: 14px;
    color: var(--text-secondary);
    font-size: 1.0em;
}
.step-badge {
    display: inline-block;
    background: var(--primary);
    color: white;
    width: 36px;
    height: 36px;
    line-height: 36px;
    text-align: center;
    border-radius: 50%;
    font-weight: bold;
    font-size: 1.1em;
    margin-right: 10px;
    vertical-align: middle;
}
.img-container {
    text-align: center;
    margin: 25px 0;
}
.img-container img {
    max-width: 100%;
    border-radius: 8px;
    border: 1px solid var(--border);
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.img-caption {
    margin-top: 10px;
    font-size: 0.92em;
    color: var(--text-secondary);
    font-style: italic;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 0.95em;
}
th {
    background: var(--primary-light);
    color: var(--primary);
    font-weight: 600;
    padding: 12px 16px;
    text-align: left;
    border-bottom: 2px solid var(--primary);
}
td {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
}
tr:hover td { background: #f8f9fa; }
code {
    background: var(--code-bg);
    padding: 2px 7px;
    border-radius: 4px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.92em;
    color: #d93025;
}
.info-box {
    background: #e8f0fe;
    border-left: 4px solid var(--primary);
    padding: 16px 20px;
    border-radius: 0 8px 8px 0;
    margin: 20px 0;
}
.info-box.warning {
    background: #fef7e0;
    border-left-color: var(--accent-orange);
}
.info-box.success {
    background: #e6f4ea;
    border-left-color: var(--accent-green);
}
.info-box strong { display: block; margin-bottom: 5px; }
.formula {
    background: var(--code-bg);
    padding: 15px 20px;
    border-radius: 8px;
    text-align: center;
    font-family: 'Cambria Math', 'Times New Roman', serif;
    font-size: 1.1em;
    margin: 15px 0;
    border: 1px solid var(--border);
}
.footer {
    text-align: center;
    padding: 30px;
    color: var(--text-secondary);
    font-size: 0.9em;
}
.category-tag {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 600;
    margin-right: 6px;
    vertical-align: middle;
}
.cat-basic { background: #e8f0fe; color: #1a73e8; }
.cat-mode { background: #fce8e6; color: #d93025; }
.cat-spatial { background: #e6f4ea; color: #1e8e3e; }
.cat-geometry { background: #fef7e0; color: #e37400; }
.cat-symmetry { background: #f3e8fd; color: #7b1fa2; }
.cat-freq { background: #e0f7fa; color: #00838f; }
.cat-mask { background: #fbe9e7; color: #bf360c; }
.cat-cross { background: #e8eaf6; color: #283593; }
.priority-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: bold;
    color: white;
}
.priority-1 { background: #d93025; }
.priority-2 { background: #e37400; }
.priority-3 { background: #1e8e3e; }
.priority-4 { background: #1a73e8; }
@media (max-width: 768px) {
    .container { padding: 10px 15px; }
    .header { padding: 30px 20px; }
    .header h1 { font-size: 1.6em; }
    .section { padding: 20px; }
}
"""


def build_html(images: dict) -> str:
    """Build the complete HTML document with embedded images."""
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PCB Warpage 특성 변수 설명서</title>
    <style>{CSS_STYLE}</style>
</head>
<body>
<div class="container">

    <!-- Header -->
    <div class="header">
        <h1>PCB Warpage 특성 변수 설명서</h1>
        <p>전처리된 PCB warpage 데이터에서 추출 가능한 변수들의 정의, 수식, 시각적 예시를 정리한 문서입니다.<br>
        공정 파라미터와의 상관분석에 활용할 수 있는 다양한 특성 변수들을 카테고리별로 소개합니다.</p>
    </div>

    <!-- Table of Contents -->
    <div class="toc">
        <h2>목차</h2>
        <ol>
            <li><a href="#overview">개요 — PCB Warpage 데이터란?</a></li>
            <li><a href="#basic-stats">기본 통계량 (Basic Statistics)</a></li>
            <li><a href="#warpage-mode">Warpage Mode (형상 분류)</a></li>
            <li><a href="#spatial">공간 분포 특성 (Spatial Distribution)</a></li>
            <li><a href="#geometry">표면 형상 특성 (Surface Geometry)</a></li>
            <li><a href="#symmetry">대칭성 특성 (Symmetry)</a></li>
            <li><a href="#frequency">주파수/스펙트럼 특성 (Frequency Domain)</a></li>
            <li><a href="#masking">마스킹 영역 특성 (Pre-interpolation)</a></li>
            <li><a href="#crosssection">Cross-section 프로파일</a></li>
            <li><a href="#summary">전체 변수 요약 테이블</a></li>
            <li><a href="#priority">추천 우선순위</a></li>
        </ol>
    </div>

    <!-- ================================================================== -->
    <!-- 1. Overview -->
    <!-- ================================================================== -->
    <div class="section" id="overview">
        <h2><span class="step-badge">1</span> 개요 — PCB Warpage 데이터란?</h2>
        <p>PCB warpage 데이터는 PCB 기판 표면의 <strong>높이(elevation)</strong>를 격자 형태로 측정한 2D 배열입니다.
        공정을 거치면서 PCB가 얼마나, 어떤 방향으로 휘었는지를 나타냅니다.</p>

        <p>전처리 파이프라인(다운샘플링 → 틸트 보정 → 이상치 제거 → 보간 → 스무딩)을 거치면,
        부품 마스킹으로 인한 빈 영역이 채워진 <strong>완전한 elevation surface</strong>가 됩니다.
        이 surface에서 다양한 특성 변수를 추출하여 공정 파라미터와의 상관관계를 분석할 수 있습니다.</p>

        <div class="img-container">
            <img src="data:image/png;base64,{images['overview']}" alt="PCB Warpage Overview">
            <div class="img-caption">그림 1. 전처리 완료된 PCB warpage 데이터의 2D heatmap(좌)과 3D surface(우) 시각화 예시.
            색상이 빨간색일수록 높은 elevation(큰 warpage), 파란색일수록 낮은 elevation을 나타냅니다.</div>
        </div>

        <div class="info-box">
            <strong>데이터 형식</strong>
            탭으로 구분된 텍스트 파일 (.txt)이며, 각 셀은 해당 위치의 elevation 값(μm 단위)입니다.
            전처리 후에는 NaN 없이 모든 셀이 유효한 값으로 채워져 있으며, 틸트 보정으로 인해 최소값은 0입니다.
        </div>
    </div>

    <!-- ================================================================== -->
    <!-- 2. Basic Statistics -->
    <!-- ================================================================== -->
    <div class="section" id="basic-stats">
        <h2><span class="step-badge">2</span> 기본 통계량 (Basic Statistics)</h2>
        <p>warpage surface의 전체적인 크기와 분포 특성을 수치로 요약하는 가장 기본적인 변수들입니다.
        계산이 간단하고 해석이 직관적이어서 공정 상관분석의 첫 번째 단계로 적합합니다.</p>

        <div class="img-container">
            <img src="data:image/png;base64,{images['basic_stats']}" alt="Basic Statistics">
            <div class="img-caption">그림 2. (좌) warpage 값의 히스토그램 — mean, max, P95, ±1σ 범위 표시.
            (우) surface heatmap에 max 위치(빨간 별)와 mean contour(흰색 점선) 표시.</div>
        </div>

        <table>
            <tr><th>변수명</th><th>수식 / 정의</th><th>설명</th></tr>
            <tr>
                <td><code>max_warpage</code></td>
                <td class="formula">max(Z)</td>
                <td>최대 warpage 값. 틸트 보정 후 min=0이므로 이 값이 곧 전체 elevation range와 같습니다.</td>
            </tr>
            <tr>
                <td><code>mean_warpage</code></td>
                <td class="formula">μ = (1/N) Σ Z<sub>i</sub></td>
                <td>평균 warpage. 전체적인 휨의 크기를 대표하는 값입니다.</td>
            </tr>
            <tr>
                <td><code>std_warpage</code></td>
                <td class="formula">σ = √[(1/N) Σ (Z<sub>i</sub> - μ)²]</td>
                <td>표준편차. 값이 크면 warpage가 불균일하게 분포되어 있다는 의미입니다.</td>
            </tr>
            <tr>
                <td><code>median_warpage</code></td>
                <td class="formula">P50(Z)</td>
                <td>중앙값. mean과 차이가 크면 분포가 비대칭적이라는 신호입니다.</td>
            </tr>
            <tr>
                <td><code>skewness</code></td>
                <td class="formula">γ = (1/N) Σ [(Z<sub>i</sub> - μ) / σ]³</td>
                <td>왜도. 양수이면 낮은 warpage 쪽으로 치우쳐 있고(대부분 낮고 일부만 높음),
                음수이면 높은 쪽으로 치우쳐 있습니다.</td>
            </tr>
            <tr>
                <td><code>kurtosis</code></td>
                <td class="formula">κ = (1/N) Σ [(Z<sub>i</sub> - μ) / σ]⁴ − 3</td>
                <td>첨도. 양수이면 극단적인 warpage가 집중되어 있고(뾰족한 분포),
                음수이면 고르게 퍼져 있습니다(평평한 분포).</td>
            </tr>
            <tr>
                <td><code>percentile_95</code></td>
                <td class="formula">P95(Z)</td>
                <td>상위 5% warpage 값. max가 이상치일 수 있으므로 보다 robust한 최대값 대안입니다.</td>
            </tr>
            <tr>
                <td><code>iqr</code></td>
                <td class="formula">P75(Z) − P25(Z)</td>
                <td>사분위 범위. warpage 분포의 중앙 50%가 차지하는 범위입니다.</td>
            </tr>
            <tr>
                <td><code>cv</code></td>
                <td class="formula">σ / μ</td>
                <td>변동계수(Coefficient of Variation). 평균 대비 상대적인 변동성을 나타냅니다.</td>
            </tr>
        </table>

        <div class="info-box success">
            <strong>공정 상관분석 활용</strong>
            max_warpage는 제품 스펙 판정의 핵심 기준이며, std_warpage는 공정 균일성을 반영합니다.
            skewness/kurtosis는 warpage 분포의 형태가 공정 조건에 따라 어떻게 달라지는지 파악하는 데 유용합니다.
        </div>
    </div>

    <!-- ================================================================== -->
    <!-- 3. Warpage Mode -->
    <!-- ================================================================== -->
    <div class="section" id="warpage-mode">
        <h2><span class="step-badge">3</span> Warpage Mode (형상 분류)</h2>
        <p>PCB가 어떤 <strong>형태</strong>로 휘었는지를 분류하는 변수들입니다.
        같은 max_warpage라도 형상에 따라 후속 공정에 미치는 영향이 크게 다르므로,
        warpage의 "크기"뿐만 아니라 "형태"를 정량화하는 것이 중요합니다.</p>

        <div class="img-container">
            <img src="data:image/png;base64,{images['warpage_mode']}" alt="Warpage Modes">
            <div class="img-caption">그림 3. PCB warpage의 4가지 대표적 형상 모드.
            상단: 2D heatmap, 하단: 중심 행(파란색)과 중심 열(빨간색) 단면 프로파일.</div>
        </div>

        <table>
            <tr><th>변수명</th><th>수식 / 정의</th><th>설명</th></tr>
            <tr>
                <td><code>center_vs_edge_ratio</code></td>
                <td class="formula">mean(Z<sub>center_region</sub>) / mean(Z<sub>edge_region</sub>)</td>
                <td>> 1이면 <strong>Crying</strong> (중심이 높음), < 1이면 <strong>Smiling</strong> (가장자리가 높음).
                center_region은 전체 면적의 중앙 25%, edge_region은 가장자리 띠 영역입니다.</td>
            </tr>
            <tr>
                <td><code>saddle_index</code></td>
                <td class="formula">(Z<sub>TL</sub> + Z<sub>BR</sub>) / 2 − (Z<sub>TR</sub> + Z<sub>BL</sub>) / 2</td>
                <td>4개 코너의 대각선 elevation 차이. 0에서 멀수록 <strong>Saddle</strong> 형상입니다.
                TL=Top-Left, BR=Bottom-Right 등 코너 영역의 평균값을 사용합니다.</td>
            </tr>
            <tr>
                <td><code>corner_elevation_std</code></td>
                <td class="formula">std([Z<sub>TL</sub>, Z<sub>TR</sub>, Z<sub>BL</sub>, Z<sub>BR</sub>])</td>
                <td>4개 코너 elevation의 표준편차. 값이 크면 <strong>Twist</strong> 가능성이 높습니다.</td>
            </tr>
            <tr>
                <td><code>warpage_mode</code></td>
                <td>위 변수들의 조합으로 분류</td>
                <td>카테고리 변수: Crying / Smiling / Saddle / Twist / Mixed.
                center_vs_edge_ratio와 saddle_index, corner_std를 종합하여 판정합니다.</td>
            </tr>
            <tr>
                <td><code>edge_profile_asymmetry</code></td>
                <td class="formula">std([mean(top), mean(bottom), mean(left), mean(right)])</td>
                <td>상/하/좌/우 4개 변(edge)의 평균 elevation 표준편차.
                값이 크면 한쪽 방향으로 비대칭적 warpage가 발생했다는 의미입니다.</td>
            </tr>
        </table>

        <div class="info-box warning">
            <strong>참고</strong>
            실제 PCB에서는 하나의 pure mode보다 여러 모드가 혼합된 경우가 대부분입니다.
            center_vs_edge_ratio와 saddle_index를 2D scatter plot으로 그려보면
            각 PCB가 어떤 형상 영역에 위치하는지 시각적으로 확인할 수 있습니다.
        </div>
    </div>

    <!-- ================================================================== -->
    <!-- 4. Spatial Distribution -->
    <!-- ================================================================== -->
    <div class="section" id="spatial">
        <h2><span class="step-badge">4</span> 공간 분포 특성 (Spatial Distribution)</h2>
        <p>warpage가 PCB 표면 위에서 <strong>어디에, 어떻게 분포</strong>하는지를 정량화하는 변수들입니다.
        같은 통계값이라도 warpage가 한쪽에 몰려있는지, 고르게 퍼져있는지에 따라 공정 원인이 다를 수 있습니다.</p>

        <div class="img-container">
            <img src="data:image/png;base64,{images['spatial']}" alt="Spatial Distribution">
            <div class="img-caption">그림 4. (좌) warpage 가중 무게중심(흰색 +)과 기하학적 중심(검은색 ×).
            (중앙) 4분면별 평균 warpage. (우) P90 이상인 고 warpage 영역(빨간색).</div>
        </div>

        <table>
            <tr><th>변수명</th><th>수식 / 정의</th><th>설명</th></tr>
            <tr>
                <td><code>max_warpage_x</code><br><code>max_warpage_y</code></td>
                <td class="formula">argmax(Z)의 (x, y) 좌표 (정규화: 0~1)</td>
                <td>최대 warpage가 발생한 위치. 특정 위치에 반복적으로 max가 나타난다면
                해당 위치의 공정 조건(온도 분포 등)을 점검할 수 있습니다.</td>
            </tr>
            <tr>
                <td><code>centroid_x</code><br><code>centroid_y</code></td>
                <td class="formula">centroid_x = Σ(x · Z) / Σ(Z)<br>centroid_y = Σ(y · Z) / Σ(Z)</td>
                <td>warpage 가중 무게중심. 기하학적 중심(0.5, 0.5)에서 멀수록
                warpage가 한쪽으로 편향되어 있다는 의미입니다.</td>
            </tr>
            <tr>
                <td><code>quadrant_means</code><br>(Q1, Q2, Q3, Q4)</td>
                <td class="formula">각 사분면 영역의 mean(Z)</td>
                <td>4분면별 평균 warpage. 분면 간 차이가 크면 비균일 공정 조건을 시사합니다.</td>
            </tr>
            <tr>
                <td><code>quadrant_max_diff</code></td>
                <td class="formula">max(Q1..Q4) − min(Q1..Q4)</td>
                <td>4분면 평균 중 최대-최소 차이. 공간적 비균일성의 단일 지표입니다.</td>
            </tr>
            <tr>
                <td><code>high_warpage_area_ratio</code></td>
                <td class="formula">count(Z ≥ P90) / N</td>
                <td>P90 이상인 픽셀의 비율. 고 warpage가 국소적인지 광범위한지를 나타냅니다.
                10%에 가까울수록 넓은 영역이 높은 warpage를 보입니다.</td>
            </tr>
            <tr>
                <td><code>spatial_entropy</code></td>
                <td class="formula">−Σ p<sub>i</sub> log(p<sub>i</sub>), p<sub>i</sub> = Z<sub>i</sub> / Σ Z</td>
                <td>공간 엔트로피. 높으면 warpage가 균일하게 퍼져있고,
                낮으면 특정 영역에 집중되어 있습니다.</td>
            </tr>
        </table>
    </div>

    <!-- ================================================================== -->
    <!-- 5. Surface Geometry -->
    <!-- ================================================================== -->
    <div class="section" id="geometry">
        <h2><span class="step-badge">5</span> 표면 형상 특성 (Surface Geometry)</h2>
        <p>warpage surface의 <strong>기울기, 곡률, 거칠기</strong> 등 미분 기반 특성입니다.
        단순한 높낮이가 아니라 표면이 얼마나 급격히 변하는지, 얼마나 매끄러운지를 정량화합니다.</p>

        <div class="img-container">
            <img src="data:image/png;base64,{images['geometry']}" alt="Surface Geometry">
            <div class="img-caption">그림 5. (1) 원본 surface, (2) gradient magnitude — 표면 기울기의 크기,
            (3) Laplacian — 곡률 (빨강=볼록, 파랑=오목), (4) roughness — smooth surface로부터의 편차.</div>
        </div>

        <table>
            <tr><th>변수명</th><th>수식 / 정의</th><th>설명</th></tr>
            <tr>
                <td><code>mean_gradient</code></td>
                <td class="formula">mean(√(∂Z/∂x)² + (∂Z/∂y)²)</td>
                <td>평균 gradient magnitude. 표면이 전체적으로 얼마나 급하게 변하는지를 나타냅니다.
                값이 크면 warpage 변화가 급격합니다.</td>
            </tr>
            <tr>
                <td><code>max_gradient</code></td>
                <td class="formula">max(√(∂Z/∂x)² + (∂Z/∂y)²)</td>
                <td>가장 급격한 변화 지점의 기울기. 응력 집중 가능 지점을 식별하는 데 유용합니다.</td>
            </tr>
            <tr>
                <td><code>mean_laplacian</code></td>
                <td class="formula">mean(∂²Z/∂x² + ∂²Z/∂y²)</td>
                <td>평균 Laplacian (곡률의 proxy). 양수이면 전체적으로 오목(concave),
                음수이면 볼록(convex)한 경향입니다.</td>
            </tr>
            <tr>
                <td><code>laplacian_std</code></td>
                <td class="formula">std(∂²Z/∂x² + ∂²Z/∂y²)</td>
                <td>Laplacian의 표준편차. 값이 크면 볼록/오목이 혼재된 복잡한 surface입니다.</td>
            </tr>
            <tr>
                <td><code>roughness_Ra</code></td>
                <td class="formula">Ra = mean(|Z − Z<sub>smooth</sub>|)</td>
                <td>산술 평균 거칠기. Gaussian smoothing된 표면으로부터의 평균 편차입니다.
                높으면 미세한 요철이 많다는 뜻입니다.</td>
            </tr>
            <tr>
                <td><code>roughness_Rq</code></td>
                <td class="formula">Rq = √[mean((Z − Z<sub>smooth</sub>)²)]</td>
                <td>RMS 거칠기. Ra보다 큰 편차에 더 민감합니다.</td>
            </tr>
        </table>

        <div class="info-box">
            <strong>gradient vs curvature</strong>
            gradient는 표면의 <em>1차 미분</em>으로, 기울어진 정도를 나타냅니다.
            Laplacian은 <em>2차 미분</em>으로, 표면이 얼마나 "굽어있는지"를 나타냅니다.
            gradient가 크더라도 일정한 기울기면 curvature는 작을 수 있습니다.
        </div>
    </div>

    <!-- ================================================================== -->
    <!-- 6. Symmetry -->
    <!-- ================================================================== -->
    <div class="section" id="symmetry">
        <h2><span class="step-badge">6</span> 대칭성 특성 (Symmetry)</h2>
        <p>PCB warpage가 <strong>대칭적인지 비대칭적인지</strong>를 측정합니다.
        대칭적 warpage는 균일한 공정 조건(온도, 압력)을 시사하며,
        비대칭적 warpage는 공정 장비의 편향이나 치구(jig) 문제를 나타낼 수 있습니다.</p>

        <div class="img-container">
            <img src="data:image/png;base64,{images['symmetry']}" alt="Symmetry Analysis">
            <div class="img-caption">그림 6. 대칭성 분석. 상단: 원본 surface와 대칭 축/점 표시.
            하단: 원본과 변환된 surface의 차이(difference map). Asymmetry 값이 작을수록 대칭적입니다.</div>
        </div>

        <table>
            <tr><th>변수명</th><th>수식 / 정의</th><th>설명</th></tr>
            <tr>
                <td><code>lr_asymmetry</code></td>
                <td class="formula">mean(|Z − flip_lr(Z)|) / range(Z)</td>
                <td>좌우 비대칭도. 0에 가까울수록 좌우 대칭.
                공정 장비의 좌우 온도 차이 등을 감지할 수 있습니다.</td>
            </tr>
            <tr>
                <td><code>tb_asymmetry</code></td>
                <td class="formula">mean(|Z − flip_ud(Z)|) / range(Z)</td>
                <td>상하 비대칭도. 0에 가까울수록 상하 대칭.
                컨베이어 방향에 따른 온도 구배를 반영할 수 있습니다.</td>
            </tr>
            <tr>
                <td><code>rotational_asymmetry_180</code></td>
                <td class="formula">mean(|Z − rot180(Z)|) / range(Z)</td>
                <td>180° 회전 비대칭도. Crying/Smiling 같은 대칭 모드는 이 값이 작고,
                Twist는 클 수 있습니다.</td>
            </tr>
        </table>

        <div class="info-box success">
            <strong>공정 상관분석 활용</strong>
            lr_asymmetry와 tb_asymmetry를 함께 보면 공정 장비의 온도 균일성을 간접 평가할 수 있습니다.
            특정 장비에서만 비대칭도가 높다면, 해당 장비의 히터나 cooling 시스템을 점검해볼 필요가 있습니다.
        </div>
    </div>

    <!-- ================================================================== -->
    <!-- 7. Frequency Domain -->
    <!-- ================================================================== -->
    <div class="section" id="frequency">
        <h2><span class="step-badge">7</span> 주파수/스펙트럼 특성 (Frequency Domain)</h2>
        <p>2D FFT(Fast Fourier Transform)를 이용하여 warpage surface를 <strong>주파수 성분</strong>으로 분해합니다.
        저주파 성분은 전체적인 굽힘(bending), 고주파 성분은 국소적 변형(local deformation)에 해당합니다.</p>

        <div class="img-container">
            <img src="data:image/png;base64,{images['frequency']}" alt="Frequency Analysis">
            <div class="img-caption">그림 7. (좌) 원본 surface. (중앙) 2D power spectrum — 중심이 저주파, 가장자리가 고주파.
            시안 원이 저주파 영역 경계. (우) 중심 행 단면의 저주파/고주파 분해.</div>
        </div>

        <table>
            <tr><th>변수명</th><th>수식 / 정의</th><th>설명</th></tr>
            <tr>
                <td><code>low_freq_energy_ratio</code></td>
                <td class="formula">Σ|F<sub>low</sub>|² / Σ|F<sub>all</sub>|²</td>
                <td>저주파 에너지 비율. 높으면 전체적 굽힘이 지배적이고,
                낮으면 국소 변형이 많다는 의미입니다.
                저주파 영역은 전체 주파수 범위의 1/8 반경 내로 정의합니다.</td>
            </tr>
            <tr>
                <td><code>dominant_freq_x</code><br><code>dominant_freq_y</code></td>
                <td class="formula">argmax(|F(k<sub>x</sub>, k<sub>y</sub>)|²) — DC 제외</td>
                <td>가장 에너지가 높은 주파수 성분의 X, Y 방향 공간 주파수.
                반복적 패턴이 있으면 특정 주파수에 에너지가 집중됩니다.</td>
            </tr>
            <tr>
                <td><code>spectral_centroid</code></td>
                <td class="formula">Σ(f · |F(f)|²) / Σ|F(f)|²</td>
                <td>스펙트럼 무게중심(주파수). 값이 작으면 저주파 지배적,
                크면 고주파 성분이 많습니다.</td>
            </tr>
        </table>

        <div class="info-box">
            <strong>저주파 vs 고주파의 물리적 의미</strong>
            <ul style="margin-top:8px; padding-left:20px;">
                <li><strong>저주파 (전체적 굽힘)</strong>: 열팽창 계수(CTE) 불일치, 기판 전체의 온도 구배 등 → 전체적 공정 조건과 관련</li>
                <li><strong>고주파 (국소 변형)</strong>: 부품 실장에 의한 국소 응력, 솔더링 패턴, via 밀집 등 → 설계/레이아웃과 관련</li>
            </ul>
        </div>
    </div>

    <!-- ================================================================== -->
    <!-- 8. Masking -->
    <!-- ================================================================== -->
    <div class="section" id="masking">
        <h2><span class="step-badge">8</span> 마스킹 영역 특성 (Pre-interpolation)</h2>
        <p>전처리 <strong>이전</strong>의 원본 데이터에서 NaN(마스킹)으로 표시된 영역에 대한 특성입니다.
        이 영역은 부품이 실장된 위치를 반영하므로, 부품 배치와 warpage의 관계를 간접적으로 분석할 수 있습니다.</p>

        <div class="img-container">
            <img src="data:image/png;base64,{images['masking']}" alt="Masking Analysis">
            <div class="img-caption">그림 8. (좌) 완전한 surface (ground truth).
            (중앙) 부품 마스킹이 적용된 surface (NaN 영역 = 흰색).
            (우) 마스킹 연결 영역 분석 — 각 색상이 하나의 connected component(부품).</div>
        </div>

        <table>
            <tr><th>변수명</th><th>수식 / 정의</th><th>설명</th></tr>
            <tr>
                <td><code>missing_ratio</code></td>
                <td class="formula">count(NaN) / N</td>
                <td>전체 픽셀 대비 NaN 비율. 부품 밀집도의 proxy이며,
                보간 품질에도 영향을 미칩니다.</td>
            </tr>
            <tr>
                <td><code>n_connected_components</code></td>
                <td class="formula">label(NaN mask)의 연결 영역 수</td>
                <td>NaN 연결 영역(connected component)의 개수.
                부품 수의 proxy입니다.</td>
            </tr>
            <tr>
                <td><code>largest_missing_area</code></td>
                <td class="formula">max(각 connected component의 픽셀 수)</td>
                <td>가장 큰 마스킹 영역의 크기(픽셀 수).
                큰 부품이 warpage에 미치는 영향을 파악할 수 있습니다.</td>
            </tr>
            <tr>
                <td><code>missing_center_ratio</code></td>
                <td class="formula">NaN_center / NaN_total</td>
                <td>NaN 영역 중 중심부에 위치한 비율.
                부품이 중심에 집중되어 있는지 가장자리에 분포하는지를 나타냅니다.</td>
            </tr>
        </table>

        <div class="info-box warning">
            <strong>주의</strong>
            이 변수들은 전처리 <strong>이전</strong>의 원본 데이터에서만 추출 가능합니다.
            전처리 완료 후에는 NaN이 보간으로 채워지므로, 원본 파일을 별도로 읽어야 합니다.
            전처리 과정에서 <code>n_interpolated</code> (보간된 픽셀 수)는 이미 기록되고 있습니다.
        </div>
    </div>

    <!-- ================================================================== -->
    <!-- 9. Cross-section -->
    <!-- ================================================================== -->
    <div class="section" id="crosssection">
        <h2><span class="step-badge">9</span> Cross-section 프로파일</h2>
        <p>surface의 특정 라인을 따라 잘라낸 <strong>1D 프로파일</strong>에서 추출하는 변수들입니다.
        2D 전체 통계만으로는 보이지 않는 방향별 warpage 패턴을 파악할 수 있습니다.</p>

        <div class="img-container">
            <img src="data:image/png;base64,{images['crosssection']}" alt="Cross-section Profiles">
            <div class="img-caption">그림 9. (좌) surface 위의 cross-section 라인 — 시안: 중심 행, 노란: 중심 열, 흰색 점선: 대각선.
            (우) 각 라인에 따른 elevation 프로파일과 range 값.</div>
        </div>

        <table>
            <tr><th>변수명</th><th>수식 / 정의</th><th>설명</th></tr>
            <tr>
                <td><code>center_row_range</code></td>
                <td class="formula">max(Z[H/2, :]) − min(Z[H/2, :])</td>
                <td>중심 행 단면의 elevation range. X 방향 warpage의 크기입니다.</td>
            </tr>
            <tr>
                <td><code>center_col_range</code></td>
                <td class="formula">max(Z[:, W/2]) − min(Z[:, W/2])</td>
                <td>중심 열 단면의 elevation range. Y 방향 warpage의 크기입니다.</td>
            </tr>
            <tr>
                <td><code>diagonal_range</code></td>
                <td class="formula">max(Z[i,i]) − min(Z[i,i])</td>
                <td>대각선 단면의 elevation range. Twist 형상에서 특히 의미가 있습니다.</td>
            </tr>
            <tr>
                <td><code>xy_range_ratio</code></td>
                <td class="formula">center_row_range / center_col_range</td>
                <td>X/Y 방향 warpage 비율. 1에서 멀수록 한 방향으로 편향된 warpage입니다.
                열 구배 방향성(thermal gradient)의 proxy가 됩니다.</td>
            </tr>
            <tr>
                <td><code>max_cross_curvature</code></td>
                <td class="formula">max(|d²Z/ds²|) along center profiles</td>
                <td>단면 프로파일의 최대 곡률. 급격한 굴곡이 있는 지점을 식별합니다.</td>
            </tr>
        </table>
    </div>

    <!-- ================================================================== -->
    <!-- 10. Summary Table -->
    <!-- ================================================================== -->
    <div class="section" id="summary">
        <h2><span class="step-badge">10</span> 전체 변수 요약 테이블</h2>
        <p>위에서 설명한 모든 변수를 한눈에 볼 수 있는 요약 테이블입니다.</p>

        <table>
            <tr>
                <th>카테고리</th>
                <th>변수명</th>
                <th>타입</th>
                <th>간단 설명</th>
            </tr>
            <!-- Basic Stats -->
            <tr><td rowspan="9"><span class="category-tag cat-basic">기본 통계</span></td>
                <td><code>max_warpage</code></td><td>연속</td><td>최대 warpage 값</td></tr>
            <tr><td><code>mean_warpage</code></td><td>연속</td><td>평균 warpage 값</td></tr>
            <tr><td><code>std_warpage</code></td><td>연속</td><td>warpage 표준편차</td></tr>
            <tr><td><code>median_warpage</code></td><td>연속</td><td>warpage 중앙값</td></tr>
            <tr><td><code>skewness</code></td><td>연속</td><td>분포 비대칭도</td></tr>
            <tr><td><code>kurtosis</code></td><td>연속</td><td>분포 첨도</td></tr>
            <tr><td><code>percentile_95</code></td><td>연속</td><td>상위 5% warpage</td></tr>
            <tr><td><code>iqr</code></td><td>연속</td><td>사분위 범위</td></tr>
            <tr><td><code>cv</code></td><td>연속</td><td>변동계수 (σ/μ)</td></tr>

            <!-- Warpage Mode -->
            <tr><td rowspan="5"><span class="category-tag cat-mode">형상 분류</span></td>
                <td><code>center_vs_edge_ratio</code></td><td>연속</td><td>중심/가장자리 비율</td></tr>
            <tr><td><code>saddle_index</code></td><td>연속</td><td>안장 형상 지수</td></tr>
            <tr><td><code>corner_elevation_std</code></td><td>연속</td><td>코너 elevation 표준편차</td></tr>
            <tr><td><code>warpage_mode</code></td><td>범주</td><td>형상 분류 (Crying/Smiling/Saddle/Twist)</td></tr>
            <tr><td><code>edge_profile_asymmetry</code></td><td>연속</td><td>변(edge) 비대칭도</td></tr>

            <!-- Spatial -->
            <tr><td rowspan="6"><span class="category-tag cat-spatial">공간 분포</span></td>
                <td><code>max_warpage_x, _y</code></td><td>연속</td><td>최대 warpage 위치</td></tr>
            <tr><td><code>centroid_x, _y</code></td><td>연속</td><td>warpage 가중 무게중심</td></tr>
            <tr><td><code>quadrant_means</code></td><td>연속×4</td><td>4분면별 평균</td></tr>
            <tr><td><code>quadrant_max_diff</code></td><td>연속</td><td>4분면 평균 차이</td></tr>
            <tr><td><code>high_warpage_area_ratio</code></td><td>연속</td><td>고 warpage 영역 비율</td></tr>
            <tr><td><code>spatial_entropy</code></td><td>연속</td><td>공간 엔트로피</td></tr>

            <!-- Geometry -->
            <tr><td rowspan="6"><span class="category-tag cat-geometry">표면 형상</span></td>
                <td><code>mean_gradient</code></td><td>연속</td><td>평균 기울기</td></tr>
            <tr><td><code>max_gradient</code></td><td>연속</td><td>최대 기울기</td></tr>
            <tr><td><code>mean_laplacian</code></td><td>연속</td><td>평균 곡률</td></tr>
            <tr><td><code>laplacian_std</code></td><td>연속</td><td>곡률 변동성</td></tr>
            <tr><td><code>roughness_Ra</code></td><td>연속</td><td>산술평균 거칠기</td></tr>
            <tr><td><code>roughness_Rq</code></td><td>연속</td><td>RMS 거칠기</td></tr>

            <!-- Symmetry -->
            <tr><td rowspan="3"><span class="category-tag cat-symmetry">대칭성</span></td>
                <td><code>lr_asymmetry</code></td><td>연속</td><td>좌우 비대칭도</td></tr>
            <tr><td><code>tb_asymmetry</code></td><td>연속</td><td>상하 비대칭도</td></tr>
            <tr><td><code>rotational_asymmetry_180</code></td><td>연속</td><td>180° 회전 비대칭도</td></tr>

            <!-- Frequency -->
            <tr><td rowspan="3"><span class="category-tag cat-freq">주파수</span></td>
                <td><code>low_freq_energy_ratio</code></td><td>연속</td><td>저주파 에너지 비율</td></tr>
            <tr><td><code>dominant_freq_x, _y</code></td><td>연속</td><td>주요 공간 주파수</td></tr>
            <tr><td><code>spectral_centroid</code></td><td>연속</td><td>스펙트럼 무게중심</td></tr>

            <!-- Masking -->
            <tr><td rowspan="4"><span class="category-tag cat-mask">마스킹</span></td>
                <td><code>missing_ratio</code></td><td>연속</td><td>NaN 비율</td></tr>
            <tr><td><code>n_connected_components</code></td><td>정수</td><td>연결 영역 수</td></tr>
            <tr><td><code>largest_missing_area</code></td><td>정수</td><td>최대 마스킹 영역 크기</td></tr>
            <tr><td><code>missing_center_ratio</code></td><td>연속</td><td>중심부 마스킹 비율</td></tr>

            <!-- Cross-section -->
            <tr><td rowspan="5"><span class="category-tag cat-cross">Cross-section</span></td>
                <td><code>center_row_range</code></td><td>연속</td><td>중심 행 단면 range</td></tr>
            <tr><td><code>center_col_range</code></td><td>연속</td><td>중심 열 단면 range</td></tr>
            <tr><td><code>diagonal_range</code></td><td>연속</td><td>대각 단면 range</td></tr>
            <tr><td><code>xy_range_ratio</code></td><td>연속</td><td>X/Y 방향 range 비율</td></tr>
            <tr><td><code>max_cross_curvature</code></td><td>연속</td><td>단면 최대 곡률</td></tr>
        </table>
    </div>

    <!-- ================================================================== -->
    <!-- 11. Priority -->
    <!-- ================================================================== -->
    <div class="section" id="priority">
        <h2><span class="step-badge">11</span> 추천 구현 우선순위</h2>
        <p>공정 상관분석 목적에 따라 단계적으로 구현하는 것을 권장합니다.</p>

        <h3><span class="priority-badge priority-1">P1</span> 최우선 — 기본 + 형상 분류</h3>
        <p>가장 해석이 쉽고, 공정 엔지니어들이 직관적으로 이해할 수 있는 변수들입니다.</p>
        <table>
            <tr><th>변수</th><th>이유</th></tr>
            <tr><td><code>max_warpage</code>, <code>mean_warpage</code>, <code>std_warpage</code></td>
                <td>제품 스펙 판정의 핵심 기준, 가장 기본적인 warpage 크기 지표</td></tr>
            <tr><td><code>skewness</code>, <code>kurtosis</code></td>
                <td>분포 형태가 공정 조건에 따라 민감하게 변화</td></tr>
            <tr><td><code>center_vs_edge_ratio</code>, <code>saddle_index</code></td>
                <td>warpage 형태를 직접 설명 — Crying/Smiling/Saddle 구분</td></tr>
        </table>

        <h3><span class="priority-badge priority-2">P2</span> 높음 — 공간 분포 + 대칭성</h3>
        <p>warpage의 공간적 패턴과 공정 균일성을 평가합니다.</p>
        <table>
            <tr><th>변수</th><th>이유</th></tr>
            <tr><td><code>centroid_x</code>, <code>centroid_y</code></td>
                <td>warpage 편향 방향 — 장비별 온도 균일성 평가</td></tr>
            <tr><td><code>quadrant_max_diff</code></td>
                <td>공간적 비균일성의 단일 지표</td></tr>
            <tr><td><code>lr_asymmetry</code>, <code>tb_asymmetry</code></td>
                <td>공정 방향성 (좌우/상하) 편향 감지</td></tr>
            <tr><td><code>xy_range_ratio</code></td>
                <td>열 구배 방향성 proxy</td></tr>
        </table>

        <h3><span class="priority-badge priority-3">P3</span> 보통 — 표면 형상 + 마스킹</h3>
        <p>보다 정밀한 surface 분석과 설계 관련 정보입니다.</p>
        <table>
            <tr><th>변수</th><th>이유</th></tr>
            <tr><td><code>mean_gradient</code>, <code>roughness_Ra</code></td>
                <td>표면 기울기와 거칠기 — 응력 분포와 관련</td></tr>
            <tr><td><code>missing_ratio</code>, <code>n_connected_components</code></td>
                <td>부품 밀집도와 warpage 관계 분석</td></tr>
        </table>

        <h3><span class="priority-badge priority-4">P4</span> 심화 — 주파수 분석</h3>
        <p>기본 변수로 충분한 상관관계가 나오지 않을 때 추가 분석에 활용합니다.</p>
        <table>
            <tr><th>변수</th><th>이유</th></tr>
            <tr><td><code>low_freq_energy_ratio</code></td>
                <td>전체 굽힘 vs 국소 변형의 비율 — 원인 분류에 유용</td></tr>
            <tr><td><code>spectral_centroid</code></td>
                <td>주파수 분포 특성 — 심화 분석용</td></tr>
        </table>
    </div>

    <!-- Footer -->
    <div class="footer">
        <p>PCB Warpage Feature Variables Documentation</p>
    </div>

</div>
</body>
</html>"""


# =============================================================================
# HTML Template V2 — Selected variables only
# =============================================================================

def build_html_v2(images: dict) -> str:
    """Build HTML document with only the selected feature variables."""
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PCB Warpage 특성 변수 설명서 (선택 변수)</title>
    <style>{CSS_STYLE}</style>
</head>
<body>
<div class="container">

    <!-- Header -->
    <div class="header">
        <h1>PCB Warpage 특성 변수 설명서</h1>
        <p>공정 상관분석을 위해 선정된 15개 핵심 변수의 정의, 수식, 시각적 예시를 정리한 문서입니다.<br>
        <code>wap_parameter_extract.py</code>로 추출되는 변수들과 1:1 대응됩니다.</p>
    </div>

    <!-- Table of Contents -->
    <div class="toc">
        <h2>목차</h2>
        <ol>
            <li><a href="#overview">개요 — PCB Warpage 데이터란?</a></li>
            <li><a href="#basic-stats">기본 통계량 (6개 변수)</a></li>
            <li><a href="#warpage-mode">Warpage Mode — 형상 분류 (5개 변수)</a></li>
            <li><a href="#geometry">표면 형상 특성 (4개 변수)</a></li>
            <li><a href="#summary">전체 변수 요약 테이블</a></li>
            <li><a href="#usage">추출 도구 사용법</a></li>
        </ol>
    </div>

    <!-- ================================================================== -->
    <!-- 1. Overview -->
    <!-- ================================================================== -->
    <div class="section" id="overview">
        <h2><span class="step-badge">1</span> 개요 — PCB Warpage 데이터란?</h2>
        <p>PCB warpage 데이터는 PCB 기판 표면의 <strong>높이(elevation)</strong>를 격자 형태로 측정한 2D 배열입니다.
        공정을 거치면서 PCB가 얼마나, 어떤 방향으로 휘었는지를 나타냅니다.</p>

        <p>전처리 파이프라인(다운샘플링 → 틸트 보정 → 이상치 제거 → 보간 → 스무딩)을 거치면,
        부품 마스킹으로 인한 빈 영역이 채워진 <strong>완전한 elevation surface</strong>가 됩니다.
        이 surface에서 다양한 특성 변수를 추출하여 공정 파라미터와의 상관관계를 분석할 수 있습니다.</p>

        <div class="img-container">
            <img src="data:image/png;base64,{images['overview']}" alt="PCB Warpage Overview">
            <div class="img-caption">그림 1. 전처리 완료된 PCB warpage 데이터의 2D heatmap(좌)과 3D surface(우) 시각화 예시.
            색상이 빨간색일수록 높은 elevation(큰 warpage), 파란색일수록 낮은 elevation을 나타냅니다.</div>
        </div>

        <div class="info-box">
            <strong>데이터 형식</strong>
            탭으로 구분된 텍스트 파일 (.txt)이며, 각 셀은 해당 위치의 elevation 값(μm 단위)입니다.
            전처리 후에는 NaN 없이 모든 셀이 유효한 값으로 채워져 있으며, 틸트 보정으로 인해 최소값은 0입니다.
        </div>

        <div class="info-box success">
            <strong>선정된 변수 카테고리 (총 15개)</strong>
            <ul style="margin-top:8px; padding-left:20px;">
                <li><span class="category-tag cat-basic">기본 통계</span> max_warpage, mean_warpage, std_warpage, median_warpage, skewness, kurtosis (6개)</li>
                <li><span class="category-tag cat-mode">형상 분류</span> center_vs_edge_ratio, saddle_index, corner_elevation_std, warpage_mode, edge_profile_asymmetry (5개)</li>
                <li><span class="category-tag cat-geometry">표면 형상</span> mean_gradient, max_gradient, mean_laplacian, laplacian_std (4개)</li>
            </ul>
        </div>
    </div>

    <!-- ================================================================== -->
    <!-- 2. Basic Statistics -->
    <!-- ================================================================== -->
    <div class="section" id="basic-stats">
        <h2><span class="step-badge">2</span> 기본 통계량 <span class="category-tag cat-basic">6개 변수</span></h2>
        <p>warpage surface의 전체적인 크기와 분포 특성을 수치로 요약하는 가장 기본적인 변수들입니다.
        계산이 간단하고 해석이 직관적이어서 공정 상관분석의 첫 번째 단계로 적합합니다.</p>

        <div class="img-container">
            <img src="data:image/png;base64,{images['basic_stats']}" alt="Basic Statistics">
            <div class="img-caption">그림 2. (좌) warpage 값의 히스토그램 — mean, max, P95, ±1σ 범위 표시.
            (우) surface heatmap에 max 위치(빨간 별)와 mean contour(흰색 점선) 표시.</div>
        </div>

        <table>
            <tr><th>변수명</th><th>수식 / 정의</th><th>설명</th></tr>
            <tr>
                <td><code>max_warpage</code></td>
                <td class="formula">max(Z)</td>
                <td>최대 warpage 값. 틸트 보정 후 min=0이므로 이 값이 곧 전체 elevation range와 같습니다.
                <strong>제품 스펙 판정의 핵심 기준</strong>이 됩니다.</td>
            </tr>
            <tr>
                <td><code>mean_warpage</code></td>
                <td class="formula">μ = (1/N) Σ Z<sub>i</sub></td>
                <td>평균 warpage. 전체적인 휨의 크기를 대표하는 값입니다.
                max가 이상치에 의해 왜곡될 수 있으므로 mean과 함께 보는 것이 중요합니다.</td>
            </tr>
            <tr>
                <td><code>std_warpage</code></td>
                <td class="formula">σ = √[(1/N) Σ (Z<sub>i</sub> - μ)²]</td>
                <td>표준편차. 값이 크면 warpage가 불균일하게 분포되어 있다는 의미입니다.
                <strong>공정 균일성</strong>을 반영하는 핵심 지표입니다.</td>
            </tr>
            <tr>
                <td><code>median_warpage</code></td>
                <td class="formula">P50(Z)</td>
                <td>중앙값. mean과 차이가 크면 분포가 비대칭적이라는 신호입니다.
                이상치에 robust한 대표값입니다.</td>
            </tr>
            <tr>
                <td><code>skewness</code></td>
                <td class="formula">γ = (1/N) Σ [(Z<sub>i</sub> - μ) / σ]³</td>
                <td>왜도. <strong>양수</strong>이면 낮은 warpage 쪽으로 치우쳐 있고(대부분 낮고 일부만 높음),
                <strong>음수</strong>이면 높은 쪽으로 치우쳐 있습니다. 0이면 대칭 분포입니다.</td>
            </tr>
            <tr>
                <td><code>kurtosis</code></td>
                <td class="formula">κ = (1/N) Σ [(Z<sub>i</sub> - μ) / σ]⁴ − 3</td>
                <td>첨도 (excess kurtosis). <strong>양수</strong>이면 극단적인 warpage가 집중되어 있고(뾰족한 분포),
                <strong>음수</strong>이면 고르게 퍼져 있습니다(평평한 분포). 0이면 정규분포와 동일합니다.</td>
            </tr>
        </table>

        <div class="info-box success">
            <strong>공정 상관분석 활용</strong>
            <ul style="margin-top:8px; padding-left:20px;">
                <li><code>max_warpage</code>: 스펙 초과 여부의 직접적 판정 기준</li>
                <li><code>std_warpage</code>: 공정 균일성 — 값이 크면 warpage 변동이 불균일</li>
                <li><code>skewness</code>: 양의 왜도가 크면 → 국소적으로 높은 warpage 발생 (hotspot)</li>
                <li><code>kurtosis</code>: 큰 양수 → 극단적 warpage 존재, 이상 공정 조건 의심</li>
            </ul>
        </div>
    </div>

    <!-- ================================================================== -->
    <!-- 3. Warpage Mode -->
    <!-- ================================================================== -->
    <div class="section" id="warpage-mode">
        <h2><span class="step-badge">3</span> Warpage Mode — 형상 분류 <span class="category-tag cat-mode">5개 변수</span></h2>
        <p>PCB가 어떤 <strong>형태</strong>로 휘었는지를 분류하는 변수들입니다.
        같은 max_warpage라도 형상에 따라 후속 공정에 미치는 영향이 크게 다르므로,
        warpage의 "크기"뿐만 아니라 "형태"를 정량화하는 것이 중요합니다.</p>

        <div class="img-container">
            <img src="data:image/png;base64,{images['warpage_mode']}" alt="Warpage Modes">
            <div class="img-caption">그림 3. PCB warpage의 4가지 대표적 형상 모드.
            상단: 2D heatmap, 하단: 중심 행(파란색)과 중심 열(빨간색) 단면 프로파일.</div>
        </div>

        <h3>형상 분류 기준</h3>
        <div class="img-container" style="margin: 15px 0;">
            <table style="margin: 0 auto; max-width: 700px;">
                <tr><th>모드</th><th>판정 조건</th><th>특징</th></tr>
                <tr><td><strong>Crying</strong> (Convex)</td>
                    <td><code>center_vs_edge_ratio > 1.05</code></td>
                    <td>중심이 높고 가장자리가 낮음 — 위로 볼록한 형태</td></tr>
                <tr><td><strong>Smiling</strong> (Concave)</td>
                    <td><code>center_vs_edge_ratio < 0.95</code></td>
                    <td>가장자리가 높고 중심이 낮음 — 아래로 오목한 형태</td></tr>
                <tr><td><strong>Saddle</strong></td>
                    <td><code>|saddle_index| > corner_std</code><br>& 둘 다 threshold 초과</td>
                    <td>대각선 방향 반대 코너가 서로 다른 높이 — 말 안장 모양</td></tr>
                <tr><td><strong>Twist</strong></td>
                    <td><code>corner_std > |saddle_index|</code><br>& 둘 다 threshold 초과</td>
                    <td>4개 코너 높이가 불규칙 — 비틀림 형태</td></tr>
                <tr><td><strong>Flat</strong></td>
                    <td>위 조건에 해당하지 않음</td>
                    <td>뚜렷한 형상 없이 비교적 평평한 상태</td></tr>
            </table>
        </div>

        <table>
            <tr><th>변수명</th><th>수식 / 정의</th><th>설명</th></tr>
            <tr>
                <td><code>center_vs_edge_ratio</code></td>
                <td class="formula">mean(Z<sub>center</sub>) / mean(Z<sub>edge</sub>)</td>
                <td>중심 영역(전체의 중앙 50%×50%)과 나머지 가장자리 영역의 평균 warpage 비율.
                > 1이면 <strong>Crying</strong>, < 1이면 <strong>Smiling</strong>.</td>
            </tr>
            <tr>
                <td><code>saddle_index</code></td>
                <td class="formula">(Z<sub>TL</sub> + Z<sub>BR</sub>)/2 − (Z<sub>TR</sub> + Z<sub>BL</sub>)/2</td>
                <td>대각선 코너 쌍의 elevation 차이.
                TL/BR은 좌상-우하, TR/BL은 우상-좌하 코너 패치 평균.
                0에서 멀수록 <strong>Saddle</strong> 형상입니다.</td>
            </tr>
            <tr>
                <td><code>corner_elevation_std</code></td>
                <td class="formula">std([Z<sub>TL</sub>, Z<sub>TR</sub>, Z<sub>BL</sub>, Z<sub>BR</sub>])</td>
                <td>4개 코너 elevation의 표준편차. 값이 크면 코너 간 높이 차이가 큰 것이며,
                <strong>Twist</strong>나 <strong>Saddle</strong> 가능성을 나타냅니다.</td>
            </tr>
            <tr>
                <td><code>warpage_mode</code></td>
                <td>위 변수들의 조합으로 분류</td>
                <td>범주형 변수: <strong>Crying / Smiling / Saddle / Twist / Flat</strong>.
                위 표의 판정 기준에 따라 자동 분류됩니다.</td>
            </tr>
            <tr>
                <td><code>edge_profile_asymmetry</code></td>
                <td class="formula">std([mean(top), mean(bottom), mean(left), mean(right)])</td>
                <td>상/하/좌/우 4개 변(edge)의 평균 elevation 표준편차.
                값이 크면 한쪽 방향으로 비대칭적 warpage가 발생했다는 의미이며,
                공정 장비의 온도 편향이나 치구(jig) 문제를 시사할 수 있습니다.</td>
            </tr>
        </table>

        <div class="info-box warning">
            <strong>참고</strong>
            실제 PCB에서는 하나의 pure mode보다 여러 모드가 혼합된 경우가 대부분입니다.
            <code>center_vs_edge_ratio</code>와 <code>saddle_index</code>를 2D scatter plot으로 그려보면
            각 PCB가 어떤 형상 영역에 위치하는지 시각적으로 확인할 수 있습니다.
        </div>
    </div>

    <!-- ================================================================== -->
    <!-- 4. Surface Geometry -->
    <!-- ================================================================== -->
    <div class="section" id="geometry">
        <h2><span class="step-badge">4</span> 표면 형상 특성 <span class="category-tag cat-geometry">4개 변수</span></h2>
        <p>warpage surface의 <strong>기울기(gradient)와 곡률(curvature)</strong>을 정량화하는 미분 기반 특성입니다.
        단순한 높낮이가 아니라, 표면이 얼마나 급격히 변하고 얼마나 굽어있는지를 나타냅니다.</p>

        <div class="img-container">
            <img src="data:image/png;base64,{images['geometry']}" alt="Surface Geometry">
            <div class="img-caption">그림 4. (1) 원본 surface, (2) gradient magnitude — 표면 기울기의 크기 (밝을수록 급경사),
            (3) Laplacian — 곡률 (빨강=볼록, 파랑=오목), (4) roughness — smooth surface로부터의 편차 (참고용).</div>
        </div>

        <table>
            <tr><th>변수명</th><th>수식 / 정의</th><th>설명</th></tr>
            <tr>
                <td><code>mean_gradient</code></td>
                <td class="formula">mean( √(∂Z/∂x)² + (∂Z/∂y)² )</td>
                <td>평균 gradient magnitude. 표면이 전체적으로 얼마나 급하게 변하는지를 나타냅니다.
                값이 크면 warpage 변화가 급격하며, <strong>응력 집중</strong> 가능성이 높습니다.</td>
            </tr>
            <tr>
                <td><code>max_gradient</code></td>
                <td class="formula">max( √(∂Z/∂x)² + (∂Z/∂y)² )</td>
                <td>가장 급격한 변화 지점의 기울기.
                이 값이 특별히 크다면 해당 위치에서 <strong>응력 집중</strong>이 발생할 수 있으며,
                크랙이나 솔더 불량의 위험 지표가 됩니다.</td>
            </tr>
            <tr>
                <td><code>mean_laplacian</code></td>
                <td class="formula">mean( ∂²Z/∂x² + ∂²Z/∂y² )</td>
                <td>평균 Laplacian (곡률의 proxy). 양수이면 전체적으로 <strong>오목(concave)</strong>,
                음수이면 <strong>볼록(convex)</strong>한 경향입니다.
                0에 가까우면 볼록/오목이 균형 잡혀 있습니다.</td>
            </tr>
            <tr>
                <td><code>laplacian_std</code></td>
                <td class="formula">std( ∂²Z/∂x² + ∂²Z/∂y² )</td>
                <td>Laplacian의 표준편차. 값이 크면 <strong>볼록/오목이 혼재</strong>된 복잡한 surface이며,
                값이 작으면 곡률이 비교적 균일합니다.</td>
            </tr>
        </table>

        <div class="info-box">
            <strong>gradient vs Laplacian (1차 미분 vs 2차 미분)</strong>
            <ul style="margin-top:8px; padding-left:20px;">
                <li><strong>Gradient (1차 미분)</strong>: 표면의 기울어진 정도. "얼마나 경사져 있는가?"</li>
                <li><strong>Laplacian (2차 미분)</strong>: 표면의 굽어진 정도. "얼마나 휘어있는가?"</li>
                <li>일정한 기울기의 경사면은 gradient는 크지만 Laplacian은 0입니다.</li>
                <li>돔(dome) 형상은 gradient도 크고 Laplacian도 큰 음수값을 가집니다.</li>
            </ul>
        </div>
    </div>

    <!-- ================================================================== -->
    <!-- 5. Summary Table -->
    <!-- ================================================================== -->
    <div class="section" id="summary">
        <h2><span class="step-badge">5</span> 전체 변수 요약 테이블</h2>
        <p>추출되는 15개 변수의 전체 목록입니다. Excel 출력 시 이 순서대로 열이 생성됩니다.</p>

        <table>
            <tr>
                <th style="width:30px;">#</th>
                <th>카테고리</th>
                <th>변수명</th>
                <th>타입</th>
                <th>간단 설명</th>
                <th>Excel 열 이름</th>
            </tr>
            <tr><td>1</td><td><span class="category-tag cat-basic">기본 통계</span></td>
                <td><code>max_warpage</code></td><td>연속</td><td>최대 warpage 값</td><td>max_warpage</td></tr>
            <tr><td>2</td><td><span class="category-tag cat-basic">기본 통계</span></td>
                <td><code>mean_warpage</code></td><td>연속</td><td>평균 warpage 값</td><td>mean_warpage</td></tr>
            <tr><td>3</td><td><span class="category-tag cat-basic">기본 통계</span></td>
                <td><code>std_warpage</code></td><td>연속</td><td>warpage 표준편차</td><td>std_warpage</td></tr>
            <tr><td>4</td><td><span class="category-tag cat-basic">기본 통계</span></td>
                <td><code>median_warpage</code></td><td>연속</td><td>warpage 중앙값</td><td>median_warpage</td></tr>
            <tr><td>5</td><td><span class="category-tag cat-basic">기본 통계</span></td>
                <td><code>skewness</code></td><td>연속</td><td>분포 비대칭도</td><td>skewness</td></tr>
            <tr><td>6</td><td><span class="category-tag cat-basic">기본 통계</span></td>
                <td><code>kurtosis</code></td><td>연속</td><td>분포 첨도</td><td>kurtosis</td></tr>
            <tr><td>7</td><td><span class="category-tag cat-mode">형상 분류</span></td>
                <td><code>center_vs_edge_ratio</code></td><td>연속</td><td>중심/가장자리 비율</td><td>center_vs_edge_ratio</td></tr>
            <tr><td>8</td><td><span class="category-tag cat-mode">형상 분류</span></td>
                <td><code>saddle_index</code></td><td>연속</td><td>안장 형상 지수</td><td>saddle_index</td></tr>
            <tr><td>9</td><td><span class="category-tag cat-mode">형상 분류</span></td>
                <td><code>corner_elevation_std</code></td><td>연속</td><td>코너 elevation 표준편차</td><td>corner_elevation_std</td></tr>
            <tr><td>10</td><td><span class="category-tag cat-mode">형상 분류</span></td>
                <td><code>warpage_mode</code></td><td>범주</td><td>형상 분류</td><td>warpage_mode</td></tr>
            <tr><td>11</td><td><span class="category-tag cat-mode">형상 분류</span></td>
                <td><code>edge_profile_asymmetry</code></td><td>연속</td><td>변(edge) 비대칭도</td><td>edge_profile_asymmetry</td></tr>
            <tr><td>12</td><td><span class="category-tag cat-geometry">표면 형상</span></td>
                <td><code>mean_gradient</code></td><td>연속</td><td>평균 기울기</td><td>mean_gradient</td></tr>
            <tr><td>13</td><td><span class="category-tag cat-geometry">표면 형상</span></td>
                <td><code>max_gradient</code></td><td>연속</td><td>최대 기울기</td><td>max_gradient</td></tr>
            <tr><td>14</td><td><span class="category-tag cat-geometry">표면 형상</span></td>
                <td><code>mean_laplacian</code></td><td>연속</td><td>평균 곡률</td><td>mean_laplacian</td></tr>
            <tr><td>15</td><td><span class="category-tag cat-geometry">표면 형상</span></td>
                <td><code>laplacian_std</code></td><td>연속</td><td>곡률 변동성</td><td>laplacian_std</td></tr>
        </table>
    </div>

    <!-- ================================================================== -->
    <!-- 6. Usage -->
    <!-- ================================================================== -->
    <div class="section" id="usage">
        <h2><span class="step-badge">6</span> 추출 도구 사용법</h2>
        <p><code>wap_parameter_extract.py</code>를 사용하여 전처리 완료된 .txt 파일에서 위 15개 변수를 자동 추출하고
        Excel 파일로 저장합니다.</p>

        <h3>설치 요구사항</h3>
        <pre>pip install numpy scipy openpyxl</pre>

        <h3>단일 파일 모드</h3>
        <pre>python preprocess/wap_parameter_extract.py --input-file /path/to/sample_preprocessed.txt</pre>
        <p>해당 파일이 위치한 폴더에 <code>warpage_features.xlsx</code>가 생성됩니다.</p>

        <h3>폴더 모드</h3>
        <pre>python preprocess/wap_parameter_extract.py --input-dir /path/to/folder</pre>
        <p>폴더 내 모든 .txt 파일을 처리하여 하나의 Excel 파일로 통합합니다.</p>

        <h3>출력 경로 지정</h3>
        <pre>python preprocess/wap_parameter_extract.py --input-dir /path/to/folder --output results.xlsx</pre>

        <h3>출력 Excel 구조</h3>
        <table>
            <tr>
                <th>pcb_name</th>
                <th>max_warpage</th>
                <th>mean_warpage</th>
                <th>...</th>
                <th>laplacian_std</th>
            </tr>
            <tr>
                <td>sample_001</td>
                <td>523.4</td>
                <td>187.2</td>
                <td>...</td>
                <td>0.8923</td>
            </tr>
            <tr>
                <td>sample_002</td>
                <td>412.1</td>
                <td>155.8</td>
                <td>...</td>
                <td>0.7156</td>
            </tr>
        </table>
        <p style="margin-top:8px;">
            첫 번째 열은 <code>pcb_name</code> (파일명에서 확장자를 뺀 이름)이며,
            이후 15개 열이 각 변수 값입니다. 카테고리별로 헤더 행이 색상으로 구분됩니다.
        </p>
    </div>

    <!-- Footer -->
    <div class="footer">
        <p>PCB Warpage Feature Variables Documentation — Selected Variables (v2)</p>
    </div>

</div>
</body>
</html>"""


# =============================================================================
# Main
# =============================================================================

def main():
    print("Generating warpage feature documentation...")
    print("  Creating synthetic surfaces and visualizations...")

    images = {}

    print("    [1/8] Overview image...")
    images['overview'] = gen_overview_image()

    print("    [2/8] Basic statistics image...")
    images['basic_stats'] = gen_basic_stats_image()

    print("    [3/8] Warpage mode image...")
    images['warpage_mode'] = gen_warpage_mode_image()

    print("    [4/8] Spatial distribution image...")
    images['spatial'] = gen_spatial_distribution_image()

    print("    [5/8] Surface geometry image...")
    images['geometry'] = gen_surface_geometry_image()

    print("    [6/8] Symmetry image...")
    images['symmetry'] = gen_symmetry_image()

    print("    [7/8] Frequency analysis image...")
    images['frequency'] = gen_frequency_image()

    print("    [8/8] Masking and cross-section images...")
    images['masking'] = gen_masking_image()
    images['crosssection'] = gen_crosssection_image()

    base_dir = os.path.dirname(__file__)

    # --- V1: Full document ---
    print("  Building HTML document (v1 - all variables)...")
    html_v1 = build_html(images)
    output_v1 = os.path.join(base_dir, "warpage_features_doc.html")
    with open(output_v1, 'w', encoding='utf-8') as f:
        f.write(html_v1)
    print(f"  [DONE] v1 saved to: {output_v1}")
    print(f"         File size: {os.path.getsize(output_v1) / 1024:.0f} KB")

    # --- V2: Selected variables only ---
    print("  Building HTML document (v2 - selected variables)...")
    html_v2 = build_html_v2(images)
    output_v2 = os.path.join(base_dir, "warpage_features_doc2.html")
    with open(output_v2, 'w', encoding='utf-8') as f:
        f.write(html_v2)
    print(f"  [DONE] v2 saved to: {output_v2}")
    print(f"         File size: {os.path.getsize(output_v2) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
