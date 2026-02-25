# PCB Elevation Data Preprocessor — How to Use

## Overview

This module batch-preprocesses PCB warpage (elevation) measurement data.
It applies noise removal, interpolation, and smoothing to produce **preprocessed text files** and **globally-scaled grayscale images**.

```
Raw .txt files  ->  Downsample  ->  Outlier Removal  ->  Polynomial Interpolation
    ->  Gaussian Smoothing  ->  Save preprocessed .txt
    ->  Global Min/Max Scaling  ->  Save grayscale .png
```

---

## Prerequisites

Install required packages:

```bash
pip install numpy scipy scikit-learn Pillow
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
python -m preprocessing.main --root-dir /path/to/root_data_dir
```

### Single-file mode — process one file

```bash
python -m preprocessing.main --input-file /path/to/sample_001.txt
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
| `--gaussian-sigma` | `1.0` | Gaussian smoothing sigma |

### Example with custom parameters

```bash
python -m preprocessing.main \
    --root-dir /path/to/data \
    --downsample-factor 4 \
    --z-threshold 2.5 \
    --poly-degree 3 \
    --ridge-alpha 0.1 \
    --gaussian-sigma 1.5
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

---

## Using from Python

You can also call the pipeline programmatically:

```python
from preprocessing.config import PreprocessorConfig
from preprocessing.main import main

config = PreprocessorConfig(
    root_dir="/path/to/data",
    downsample_factor=4,
    gaussian_sigma=1.0,
)
main(config)
```

Or process individual steps:

```python
from preprocessing.io_utils import read_elevation
from preprocessing.preprocessing import (
    downsample_median,
    detect_and_remove_outliers,
    interpolate_surface,
    smooth_gaussian,
)

data = read_elevation("sample.txt", null_value=9999.0)
data = downsample_median(data, factor=4)
data = detect_and_remove_outliers(data, grid_size=8, z_threshold=3.0)
data = interpolate_surface(data, poly_degree=3, ridge_alpha=0.1)
data = smooth_gaussian(data, sigma=1.0)
```

---

## Pipeline Steps

| Step | Function | Description |
|---|---|---|
| 1 | `read_elevation()` | Load `.txt`, replace 9999.0 with NaN |
| 2 | `downsample_median()` | Reduce resolution via block median |
| 3 | `detect_and_remove_outliers()` | Regional z-score outlier removal |
| 4 | `interpolate_surface()` | Polynomial + Ridge regression to fill NaN |
| 5 | `smooth_gaussian()` | Gaussian filter for smooth surface |
| 6 | `compute_global_minmax()` | Find min/max across all preprocessed files |
| 7 | `generate_grayscale_image()` | Min-max scale to [0, 255] and save PNG |

---

## Module Structure

```
preprocessing/
├── __init__.py          # Package init
├── config.py            # PreprocessorConfig dataclass
├── io_utils.py          # File I/O, discovery, skip logic
├── preprocessing.py     # Core processing functions (steps 2-5)
├── imaging.py           # Global scaling & image generation (steps 6-7)
├── main.py              # CLI entry point & orchestration
└── how_to_use.md        # This file
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: sklearn` | Run `pip install scikit-learn` |
| No output files generated | Check that subfolders exist directly under `--root-dir` and contain `.txt` files |
| All images look the same shade | If `global_min == global_max`, all pixels become 128 (mid-gray). Check input data. |
| Warnings about low valid-value ratio | Many NaN/null values in input. Interpolation quality may be poor — consider adjusting `--downsample-factor` or checking raw data. |
