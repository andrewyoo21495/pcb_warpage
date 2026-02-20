#!/usr/bin/env python3
"""Analyse handcrafted features and select the most informative subset.

Selection strategy (three complementary criteria):
  1. Variance    — features that barely change across designs are useless.
  2. Redundancy  — among highly correlated features, keep only one.
  3. Relevance   — if elevation data exists, rank by correlation with
                   per-design elevation statistics (mean warpage, std).

Usage:
  python analyze_features.py                      # print ranking only
  python analyze_features.py --top 10             # print top-10
  python analyze_features.py --top 10 --write     # also write to config.txt

The script writes a comma-separated list of selected feature **indices** to
the config key ``selected_features``.  When this key is present the design
encoder will use only those features.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from utils.handcrafted_features import extract_handcrafted_features, HAND_FEATURE_DIM


# Feature labels (must match order in extract_handcrafted_features)
FEATURE_LABELS = [
    'global_density',      # 0
    'quadrant_TL',         # 1
    'quadrant_TR',         # 2
    'quadrant_BL',         # 3
    'quadrant_BR',         # 4
    'lr_asymmetry',        # 5
    'tb_asymmetry',        # 6
    'h_line_ratio',        # 7
    'v_line_ratio',        # 8
    'd_line_ratio',        # 9
    'aspect_ratio',        # 10
    'log_num_components',  # 11
    'mean_component_area', # 12
    'lr_symmetry',         # 13
    'tb_symmetry',         # 14
    'rot180_symmetry',     # 15
    'centre_density',      # 16
    'border_density',      # 17
    'radial_mean',         # 18
    'perimeter_ratio',     # 19
    'hu_moment_1',         # 20
    'hu_moment_2',         # 21
]


def _load_design_images(design_dir: Path) -> list[tuple[str, Image.Image]]:
    """Load all design PNG files from *design_dir*."""
    paths = sorted(design_dir.glob('*.png'))
    if not paths:
        raise FileNotFoundError(f"No PNG files found in {design_dir}")
    images = []
    for p in paths:
        images.append((p.stem, Image.open(p).convert('L')))
    return images


def _extract_all(designs: list[tuple[str, Image.Image]]) -> np.ndarray:
    """Return (N_designs, HAND_FEATURE_DIM) feature matrix."""
    feats = []
    for _name, img in designs:
        f = extract_handcrafted_features(img).numpy()
        feats.append(f)
    return np.array(feats)


def _load_elevation_stats(
    elevation_dir: Path,
    design_names: list[str],
) -> np.ndarray | None:
    """Compute per-design elevation statistics: [mean, std].

    Returns (N_designs, 2) or None if elevation data is unavailable.
    """
    stats = []
    for name in design_names:
        elev_dir = elevation_dir / name
        if not elev_dir.exists():
            return None
        pngs = sorted(elev_dir.glob('*.png'))
        if not pngs:
            return None
        means, stds = [], []
        for p in pngs:
            arr = np.array(Image.open(p).convert('L'), dtype=np.float32) / 255.0
            means.append(arr.mean())
            stds.append(arr.std())
        stats.append([float(np.mean(means)), float(np.mean(stds))])
    return np.array(stats)


def rank_features(
    feat_matrix: np.ndarray,
    elev_stats: np.ndarray | None = None,
    corr_threshold: float = 0.90,
) -> list[tuple[int, str, float]]:
    """Rank features by combined importance score.

    Returns list of (index, label, score) sorted descending by score.

    Scoring:
      - normalised_variance  (0–1)  : higher = more discriminative
      - relevance            (0–1)  : |correlation| with elevation stats
      - redundancy_penalty   (0–1)  : penalise features highly correlated
                                      with a higher-ranked feature

    Final score = 0.4 * variance + 0.4 * relevance + 0.2 * (1 - redundancy)
    When elevation data is absent, relevance weight is redistributed to variance.
    """
    N, D = feat_matrix.shape

    # --- 1. Variance score ---
    variances = feat_matrix.var(axis=0)
    var_max = variances.max() if variances.max() > 0 else 1.0
    var_score = variances / var_max  # normalise to [0, 1]

    # --- 2. Relevance score (correlation with elevation stats) ---
    if elev_stats is not None and N >= 3:
        # Correlate each feature with elevation mean and std
        rel_scores = np.zeros(D)
        for j in range(D):
            col = feat_matrix[:, j]
            if col.std() < 1e-10:
                continue
            for s in range(elev_stats.shape[1]):
                target = elev_stats[:, s]
                if target.std() < 1e-10:
                    continue
                r = np.corrcoef(col, target)[0, 1]
                rel_scores[j] = max(rel_scores[j], abs(r))
        has_relevance = True
    else:
        rel_scores = np.zeros(D)
        has_relevance = False

    # --- 3. Redundancy penalty (pairwise correlation among features) ---
    corr_matrix = np.corrcoef(feat_matrix.T)  # (D, D)
    # Replace NaN (from zero-variance features) with 0
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 0.0)
    # For each feature, redundancy = max |corr| with any other feature
    redundancy = np.max(np.abs(corr_matrix), axis=1)
    # Clip to [0, 1]
    redundancy = np.clip(redundancy, 0.0, 1.0)

    # --- Combined score ---
    if has_relevance:
        score = 0.4 * var_score + 0.4 * rel_scores + 0.2 * (1.0 - redundancy)
    else:
        score = 0.7 * var_score + 0.3 * (1.0 - redundancy)

    # Build ranked list
    ranked = [(int(i), FEATURE_LABELS[i], float(score[i])) for i in range(D)]
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked


def select_top_k(
    ranked: list[tuple[int, str, float]],
    feat_matrix: np.ndarray,
    k: int = 10,
    corr_threshold: float = 0.85,
) -> list[int]:
    """Greedily select top-k features, skipping highly redundant ones.

    Walks the ranked list top-down.  A feature is added only if its
    absolute correlation with every already-selected feature is below
    *corr_threshold*.  Continues until *k* features are selected or
    the list is exhausted.
    """
    selected: list[int] = []
    for idx, _label, _score in ranked:
        if len(selected) >= k:
            break
        col = feat_matrix[:, idx]
        if col.std() < 1e-10:
            continue  # skip constant features
        redundant = False
        for sel_idx in selected:
            r = abs(np.corrcoef(col, feat_matrix[:, sel_idx])[0, 1])
            if r >= corr_threshold:
                redundant = True
                break
        if not redundant:
            selected.append(idx)
    return sorted(selected)


def write_config_key(config_path: Path, key: str, value: str) -> None:
    """Add or update a key in config.txt."""
    lines = config_path.read_text(encoding='utf-8').splitlines(keepends=True)
    found = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and not stripped.startswith('%'):
            parts = stripped.split(None, 1)
            if parts and parts[0].lower() == key.lower():
                new_lines.append(f"{key}  {value}\n")
                found = True
                continue
        new_lines.append(line)
    if not found:
        # Insert before the Inference section or at end
        new_lines.append(f"\n{key}  {value}\n")
    config_path.write_text(''.join(new_lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Analyse and select handcrafted features')
    parser.add_argument('--design_dir', type=str, default='./data/design',
                        help='Directory containing design PNGs')
    parser.add_argument('--elevation_dir', type=str, default='./data/elevation',
                        help='Directory containing elevation subfolders')
    parser.add_argument('--top', type=int, default=HAND_FEATURE_DIM,
                        help='Number of features to select (default: all)')
    parser.add_argument('--corr_threshold', type=float, default=0.85,
                        help='Max pairwise |correlation| between selected features')
    parser.add_argument('--write', action='store_true',
                        help='Write selected_features to config.txt')
    parser.add_argument('--config', type=str, default='./config.txt',
                        help='Path to config.txt')
    args = parser.parse_args()

    design_dir = Path(args.design_dir)
    elevation_dir = Path(args.elevation_dir)

    # Load designs
    print(f"Loading designs from {design_dir} ...")
    designs = _load_design_images(design_dir)
    print(f"  Found {len(designs)} designs: {[n for n, _ in designs]}")

    # Extract features
    feat_matrix = _extract_all(designs)
    print(f"  Feature matrix: {feat_matrix.shape}")
    print()

    # Print feature values per design
    print("Feature values per design:")
    print(f"  {'Feature':<22s}", end='')
    for name, _ in designs:
        print(f"  {name:>10s}", end='')
    print()
    print("  " + "-" * (22 + 12 * len(designs)))
    for j in range(HAND_FEATURE_DIM):
        print(f"  [{j:2d}] {FEATURE_LABELS[j]:<17s}", end='')
        for i in range(len(designs)):
            print(f"  {feat_matrix[i, j]:10.4f}", end='')
        print()
    print()

    # Load elevation stats if available
    design_names = [n for n, _ in designs]
    elev_stats = _load_elevation_stats(elevation_dir, design_names)
    if elev_stats is not None:
        print("Elevation statistics (per design):")
        for i, name in enumerate(design_names):
            print(f"  {name}: mean={elev_stats[i, 0]:.4f}, std={elev_stats[i, 1]:.4f}")
        print()
    else:
        print("Elevation data not found — ranking by variance and redundancy only.\n")

    # Rank features
    ranked = rank_features(feat_matrix, elev_stats)
    print("Feature ranking (descending by importance score):")
    print(f"  {'Rank':>4s}  {'Idx':>3s}  {'Feature':<22s}  {'Score':>6s}")
    print("  " + "-" * 42)
    for rank, (idx, label, score) in enumerate(ranked, 1):
        print(f"  {rank:4d}  [{idx:2d}]  {label:<22s}  {score:6.3f}")
    print()

    # Select top-k
    selected = select_top_k(ranked, feat_matrix, k=args.top,
                            corr_threshold=args.corr_threshold)
    print(f"Selected {len(selected)} features (corr_threshold={args.corr_threshold}):")
    for idx in selected:
        print(f"  [{idx:2d}] {FEATURE_LABELS[idx]}")
    indices_str = ','.join(str(i) for i in selected)
    print(f"\nConfig value:  selected_features  {indices_str}")

    # Write to config
    if args.write:
        config_path = Path(args.config)
        write_config_key(config_path, 'selected_features', indices_str)
        print(f"  → Written to {config_path}")


if __name__ == '__main__':
    main()
