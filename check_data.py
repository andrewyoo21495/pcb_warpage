#!/usr/bin/env python3
"""Diagnostic script to inspect elevation PNG pixel statistics.

Run:
    python check_data.py
"""

import numpy as np
from PIL import Image
from pathlib import Path

from utils.load_config import load_config

config = load_config('config.txt')

elevation_base_dir = Path(config.get('elevation_base_dir', 'data/elevation'))
elevation_subdir   = str(config.get('elevation_subdir', 'images')).strip()
design_names_raw   = config.get('design_names', '')
if isinstance(design_names_raw, list):
    design_names = [s.strip() for s in design_names_raw]
else:
    design_names = [s.strip() for s in str(design_names_raw).split(',')]

print(f"Elevation base dir : {elevation_base_dir}")
print(f"Designs            : {design_names}")
print()

all_means = []

for design in design_names:
    if elevation_subdir:
        elev_dir = elevation_base_dir / design / elevation_subdir
    else:
        elev_dir = elevation_base_dir / design

    pngs = sorted(elev_dir.glob('*.png'))
    if not pngs:
        print(f"[{design}] No PNG files found in {elev_dir}")
        continue

    sample_paths = pngs[:5]
    design_means = []

    print(f"[{design}]  ({len(pngs)} total files)")
    for p in sample_paths:
        img = Image.open(p)
        arr = np.array(img)
        mean_val = arr.mean()
        design_means.append(mean_val)
        print(f"  {p.name}: mode={img.mode}  dtype={arr.dtype}  "
              f"min={arr.min()}  max={arr.max()}  mean={mean_val:.1f}  "
              f"-> [0,1] mean={mean_val / arr.max() if arr.max() > 0 else 0:.3f}")

    design_mean = float(np.mean(design_means))
    all_means.append(design_mean)
    print(f"  => sample mean (uint): {design_mean:.1f}  "
          f"normalized /255: {design_mean/255:.3f}\n")

if all_means:
    overall = float(np.mean(all_means))
    print("=" * 60)
    print(f"Overall mean across designs (uint8 pixel): {overall:.1f}")
    print(f"Normalized /255                          : {overall/255:.3f}")
    print()
    if overall / 255 < 0.25:
        print(">> VALUES ARE LOW: physical data likely concentrated in lower")
        print("   portion of the normalization range (e.g. max set to 3000")
        print("   but actual values are much smaller).")
    elif overall / 255 > 0.35:
        print(">> VALUES LOOK NORMAL: if histogram still shows ~0.16,")
        print("   there may be a double-normalization issue in the pipeline.")
