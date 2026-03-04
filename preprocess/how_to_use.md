# PCB Elevation Data Preprocessor — How to Use

## Overview

This module batch-preprocesses PCB warpage (elevation) measurement data.
It applies noise removal, interpolation, and smoothing to produce **preprocessed text files** and **globally-scaled grayscale images**.

```
Raw .txt files  ->  Downsample  ->  Outlier Removal  ->  Polynomial Interpolation
    ->  Tilt Correction  ->  Gaussian Smoothing  ->  Save preprocessed .txt
    ->  Global Min/Max Scaling  ->  Elevation Range Analysis  ->  Save grayscale .png
```

---

## Prerequisites

Install required packages:

```bash
pip install numpy scipy scikit-learn Pillow matplotlib
```

---

## Input Data Format

- **Tab-delimited `.txt` files** containing an N x M numeric matrix (float).
- `9999.0` is treated as the null/missing value.
- Files whose name (without `.txt`) ends with `ORI`, `ORI@LOW`, or `ORI_A` are automatically skipped.

### Expected directory structure (batch mode)

```
root_data_dir/
├── board_A/
│   ├── sample_001.txt
│   ├── sample_002.txt
│   └── sample_003_ORI.txt    <- skipped
├── board_B/
│   ├── sample_010.txt
│   └── sample_011.txt
└── ...
```

---

## Usage

Run from the **project root** (`pcb_warpage/`).

### Batch mode — process all subfolders

```bash
python -m preprocess.main --root-dir /path/to/root_data_dir
```

### Single-file mode — process one file

```bash
python -m preprocess.main --input-file /path/to/sample_001.txt
```

> `--root-dir` and `--input-file` are mutually exclusive.

### Optional parameters

| Flag | Default | Description |
|---|---|---|
| `--downsample-factor` | `4` | Downsampling ratio (e.g., 4 = 1/4 resolution) |
| `--z-threshold` | `3.0` | Z-score threshold for outlier detection |
| `--grid-size` | `8` | Grid divisions for regional outlier detection (8 = 8x8 regions) |
| `--poly-degree` | `3` | Polynomial degree for surface interpolation |
| `--ridge-alpha` | `0.1` | Ridge regularization coefficient |
| `--tilt-patch-size` | `16` | Corner patch size (pixels) for tilt correction plane fitting |
| `--gaussian-sigma` | `2.0` | Gaussian smoothing base sigma (adaptive to data dimensions) |
| `--smooth-iterations` | `3` | Number of smooth-then-rescale iterations |
| `--workers` | `1` | Number of parallel workers (1 = sequential) |

### Example with custom parameters

```bash
python -m preprocess.main \
    --root-dir /path/to/data \
    --downsample-factor 4 \
    --z-threshold 2.5 \
    --poly-degree 3 \
    --ridge-alpha 0.1 \
    --tilt-patch-size 16 \
    --gaussian-sigma 2.0 \
    --smooth-iterations 3 \
    --workers 4
```

---

## Step-by-Step Visualization

To visually inspect each preprocessing stage on a single file, use `visualize_steps.py`:

```bash
python -m preprocess.visualize_steps --input-file /path/to/sample_001.txt
```

This displays a 2x4 figure with:

| | | | |
|---|---|---|---|
| 1. Original (jet colormap) | 2. Downsampled | 3. Outliers (red circles) | 4. Interpolated |
| 5. Smoothed | 6. Grayscale | 7. Tilt Corrected (grayscale) | 8. Zero Aligned (grayscale) |

To save the figure:

```bash
python -m preprocess.visualize_steps --input-file /path/to/sample.txt --output steps.png
```

All preprocessing parameters (`--downsample-factor`, `--z-threshold`, etc.) are accepted.

---

## Standalone Script

A consolidated single-file version is also available at `preprocess_total.py`. It contains all functionality in one file and can be run directly without package imports:

```bash
python preprocess/preprocess_total.py --root-dir /path/to/data
python preprocess/preprocess_total.py --input-file /path/to/sample.txt
```

---

## Output

After processing, outputs are placed **inside each subfolder**:

```
root_data_dir/
├── board_A/
│   ├── interpolated/
│   │   ├── sample_001_preprocessed.txt    <- tab-delimited, 4 decimal places
│   │   └── sample_002_preprocessed.txt
│   ├── images/
│   │   ├── sample_001_preprocessed.png    <- 8-bit grayscale
│   │   └── sample_002_preprocessed.png
│   ├── sample_001.txt                     <- original (unchanged)
│   └── sample_002.txt
└── ...
```

- **`interpolated/`** — preprocessed numeric data (`.txt`).
- **`images/`** — grayscale images scaled to a **global** min/max across all files (batch mode) or the file's own range (single-file mode).

### Terminal output

The pipeline displays:
- Per-file progress counter
- Total outliers removed and points interpolated
- Average tilt plane amplitude (how much tilt was removed)
- Per-subfolder min/max statistics
- Elevation range distribution table (per-subfolder and global): mean, std, min, P25, P50, P75, max of per-file (max - min) ranges
- Per-subfolder range distribution histograms saved to `outputs/distribution/`
- Global min/max values used for image scaling

---

## Using from Python

You can also call the pipeline programmatically:

```python
from preprocess.config import PreprocessorConfig
from preprocess.main import main

config = PreprocessorConfig(
    root_dir="/path/to/data",
    downsample_factor=4,
    tilt_patch_size=16,
    gaussian_sigma=2.0,
    gaussian_iterations=3,
)
main(config)
```

Or process individual steps:

```python
from preprocess.io_utils import read_elevation
from preprocess.preprocessing import (
    downsample_median,
    detect_and_remove_outliers,
    interpolate_surface,
    flatten_tilt,
    smooth_gaussian,
)

data = read_elevation("sample.txt", null_value=9999.0)
data = downsample_median(data, factor=4)
data, n_outliers = detect_and_remove_outliers(data, grid_size=8, z_threshold=3.0)
data, n_interpolated = interpolate_surface(data, poly_degree=3, ridge_alpha=0.1)
data, plane_amplitude = flatten_tilt(data, patch_size=16)
data = smooth_gaussian(data, sigma=2.0, iterations=3)
```

---

## Pipeline Steps

| Step | Function | Description |
|---|---|---|
| 1 | `read_elevation()` | Load `.txt`, replace 9999.0 with NaN |
| 2 | `downsample_median()` | Reduce resolution via block median |
| 3 | `detect_and_remove_outliers()` | Regional z-score outlier removal (returns count) |
| 4 | `interpolate_surface()` | Polynomial + Ridge regression to fill NaN (returns count) |
| 5 | `flatten_tilt()` | Corner-patch plane subtraction to remove linear tilt (returns plane amplitude) |
| 6 | `smooth_gaussian()` | Adaptive anisotropic Gaussian with min/max preservation |
| 7 | Global min/max computation | Find min/max across all preprocessed files |
| 7a | Elevation range analysis | Per-subfolder and global distribution of per-file (max - min) ranges |
| 7b | Range distribution histograms | Save per-subfolder histograms to `outputs/distribution/` |
| 8 | `generate_grayscale_image()` | Min-max scale to [0, 255] and save PNG |

---

## Tilt Correction (Step 5)

PCB elevation data often includes a linear tilt component caused by uneven component mounting, measurement fixture misalignment, or gravitational sag. This tilt is not part of the actual warpage pattern and can distort both training and analysis.

The `flatten_tilt()` function removes this linear tilt by:

1. **Corner patch averaging** — Computes the mean elevation over a square patch (default 16x16 pixels) at each of the four corners. Averaging over a patch rather than using single corner pixels makes the estimate robust to local noise.
2. **Plane fitting** — Fits a least-squares plane `z = a*row + b*col + c` through the four corner centroids. With 4 points and 3 unknowns, the system is overdetermined by one degree, providing a stable fit.
3. **Plane subtraction** — Subtracts the fitted plane from the entire surface, then shifts the result so the minimum is zero.

The function returns a `plane_amplitude` diagnostic (max - min of the subtracted plane), which indicates how much tilt was removed. Large values suggest the sample had significant tilt; near-zero values mean the sample was already level.

**Placement rationale:** Tilt correction runs after interpolation (which fills NaN gaps, ensuring complete corner data) and before Gaussian smoothing (so smoothing preserves the true warpage extremes rather than tilt-inflated ones).

Adjust the patch size with `--tilt-patch-size`. Larger patches are more noise-robust but may average over real warpage features near corners.

---

## Elevation Range Distribution Analysis (Step 7a)

In batch mode, the pipeline computes the **per-file elevation range** (max - min) for every preprocessed file and reports the distribution of these ranges both per subfolder and globally.

### Console output

A summary table is printed showing mean, std, min, P25, P50 (median), P75, and max of the per-file ranges:

```
    Elevation range distributions (per-file max - min):
    ----------------------------------------------------------------------------
      Subfolder                           Mean       Std       Min       P25       P50       P75       Max
    ----------------------------------------------------------------------------
      board_A                           0.3842    0.0521    0.2710    0.3481    0.3820    0.4189    0.5103
      board_B                           0.4215    0.0388    0.3520    0.3941    0.4200    0.4482    0.5210
    ----------------------------------------------------------------------------
      GLOBAL                            0.4029    0.0489    0.2710    0.3612    0.3950    0.4301    0.5210
```

### Saved metadata

The range distribution statistics are also saved in `scaling_metadata.json` under each subfolder entry (`range_distribution`) and at the top level (`global_range_distribution`), with fields: `mean`, `std`, `min`, `p25`, `median`, `p75`, `max`.

### Distribution histograms

In batch mode, histograms of per-file elevation ranges are automatically saved as PNG files:

```
outputs/distribution/
├── dist_board_A.png
├── dist_board_B.png
└── ...
```

Each histogram shows the distribution of per-file (max - min) ranges for that subfolder, with a red dashed line marking the mean. The naming convention is `dist_{folder_name}.png`.

This analysis is useful for:
- Detecting subfolders with unusually narrow or wide warpage ranges
- Comparing range distributions before and after tilt correction
- Identifying outlier samples with extreme ranges

---

## Module Structure

```
preprocessing/
├── __init__.py              # Package init
├── config.py                # PreprocessorConfig dataclass
├── io_utils.py              # File I/O, discovery, skip logic
├── preprocessing.py         # Core processing functions (steps 2-6)
├── imaging.py               # Global scaling & image generation (steps 7-8)
├── main.py                  # CLI entry point & orchestration
├── visualize_steps.py       # Step-by-step visualization tool
├── preprocess_total.py      # Standalone single-file version
└── how_to_use.md            # This file
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: sklearn` | Run `pip install scikit-learn` |
| `ModuleNotFoundError: matplotlib` | Run `pip install matplotlib` (needed for `visualize_steps.py`) |
| No output files generated | Check that subfolders exist directly under `--root-dir` and contain `.txt` files |
| All images look the same shade | If `global_min == global_max`, all pixels become 128 (mid-gray). Check input data. |
| Warnings about low valid-value ratio | Many NaN/null values in input. Interpolation quality may be poor — consider adjusting `--downsample-factor` or checking raw data. |
