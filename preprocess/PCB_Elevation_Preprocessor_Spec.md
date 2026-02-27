# PCB Elevation Data Preprocessor — System Specification

## 1. Overview

A pipeline that batch-preprocesses PCB warpage (elevation) measurement data — applying noise removal, interpolation, and smoothing — to produce **preprocessed text files** and **globally-scaled grayscale images**.

```
[Raw .txt files in subfolders]
    ↓  1. Read
    ↓  2. Downsample (median)
    ↓  3. Outlier Detection (z-score per 8×8 region)
    ↓  4. Polynomial Interpolation (degree=3, ridge α=0.1)
    ↓  5. Gaussian Smoothing
    ↓  → Save *_preprocessed.txt  →  {subfolder}/interpolated/
    ↓
[Global min/max across ALL preprocessed files]
    ↓  6. Min-Max Scaling
    ↓  7. Grayscale Image Generation
    ↓  → Save *_preprocessed.png  →  {subfolder}/images/
```

---

## 2. Directory Structure

### 2.1 Input

```
root_input_dir/
├── subfolder_A/
│   ├── sample_001.txt
│   ├── sample_002.txt
│   └── ...
├── subfolder_B/
│   ├── sample_010.txt
│   └── ...
└── ...
```

- Each subfolder is assumed to contain measurement files from the same condition (or the same board).
- `.txt` files: tab-delimited, N×M numeric matrix (float). `9999.0` = null.
- **Exclusion rule:** `.txt` files whose filename (excluding extension) ends with any of the following suffixes are excluded from processing:
  - `ORI`
  - `ORI@LOW`
  - `ORI_A`
  - Examples: `sample_001_ORI.txt`, `board_ORI@LOW.txt`, `data_ORI_A.txt` → skipped

### 2.2 Output

```
root_input_dir/
├── subfolder_A/
│   ├── interpolated/
│   │   ├── sample_001_preprocessed.txt
│   │   └── sample_002_preprocessed.txt
│   ├── images/
│   │   ├── sample_001_preprocessed.png
│   │   └── sample_002_preprocessed.png
│   ├── sample_001.txt          # (original preserved)
│   └── sample_002.txt
└── ...
```

---

## 3. Configuration

All parameters are managed through a single config dict (or dataclass).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `root_dir` | `str \| None` | `None` | Root input directory path (batch mode) |
| `input_file` | `str \| None` | `None` | Single txt file path (single-file mode) |
| `exclude_suffixes` | `list[str]` | `["ORI", "ORI@LOW", "ORI_A"]` | Skip files whose name ends with these suffixes |
| `null_value` | `float` | `9999.0` | Sentinel value representing null |
| `downsample_factor` | `int` | `4` | 1/n downsampling ratio (e.g., 4 → 1/4) |
| `outlier_z_threshold` | `float` | `3.0` | Z-score threshold for outlier detection |
| `outlier_grid_size` | `int` | `8` | Number of grid divisions for outlier detection (8 → 8×8 = 64 regions) |
| `interp_poly_degree` | `int` | `3` | Polynomial interpolation degree |
| `interp_ridge_alpha` | `float` | `0.1` | Ridge regularization coefficient |
| `gaussian_sigma` | `float` | `1.0` | Gaussian smoothing σ (adjustable) |
| `image_format` | `str` | `"png"` | Output image format |
| `colormap` | `str` | `"gray"` | Matplotlib colormap name |

---

## 4. Processing Pipeline Detail

### 4.1 Step 1 — Read Original Data

```python
def read_elevation(filepath: str, null_value: float = 9999.0) -> np.ndarray:
```

- Load via `np.loadtxt(filepath, delimiter='\t')`.
- Replace `9999.0` values with `np.nan` for unified internal handling.
- Returns: `np.ndarray` (shape: N×M, dtype: float64, NaN = missing).

```python
EXCLUDE_SUFFIXES = ("ORI", "ORI@LOW", "ORI_A")

def should_skip_file(filepath: str, exclude_suffixes: list[str] = EXCLUDE_SUFFIXES) -> bool:
    """Returns True if the filename (without extension) ends with any of the exclude_suffixes."""
```

- Strips the extension first, then checks via `str.endswith()`.
- Example: `"board_01_ORI.txt"` → stem `"board_01_ORI"` → `endswith("ORI")` → **skip**.

### 4.2 Step 2 — Downsampling (Median)

```python
def downsample_median(data: np.ndarray, factor: int) -> np.ndarray:
```

**Purpose:** Reduce original resolution to `1/factor` while performing first-pass noise filtering.

**Algorithm:**
1. Original shape `(H, W)` → target shape `(H // factor, W // factor)`.
2. Partition the original into `factor × factor` blocks.
3. Compute the median of **non-NaN** values in each block.
   - If a block contains no valid values, retain `np.nan`.
4. If the original dimensions are not exactly divisible by `factor`, **truncate** the remainder rows/columns.

**Implementation Notes:**
- Use `np.nanmedian`.
- Block partitioning via reshape + axis operations or explicit loops (prioritize correctness over performance).

### 4.3 Step 3 — Outlier Detection (Regional Z-Score)

```python
def detect_and_remove_outliers(
    data: np.ndarray,
    grid_size: int = 8,
    z_threshold: float = 3.0
) -> np.ndarray:
```

**Purpose:** Detect outliers on a per-region basis and replace them with null.

**Algorithm:**
1. Divide the downsampled data `(H, W)` into `grid_size × grid_size` regions (64 total regions).
   - Region size: `(H // grid_size, W // grid_size)` — remainder pixels are included in the last region.
2. For each region:
   a. Extract non-NaN valid values.
   b. Compute mean and std.
   c. Flag values where `|value - mean| / std > z_threshold` as outliers.
   d. Replace outliers with `np.nan`.
3. Return the fully processed result.

**Edge Cases:**
- If a region has fewer than 2 valid values, skip outlier detection (std cannot be computed).
- If std = 0 (all values identical), treat as no outliers.

### 4.4 Step 4 — Polynomial Interpolation with Ridge Regularization

```python
def interpolate_surface(
    data: np.ndarray,
    poly_degree: int = 3,
    ridge_alpha: float = 0.1
) -> np.ndarray:
```

**Purpose:** Fill NaN (missing) values using a polynomial surface fitted to the valid data points.

**Algorithm:**
1. Extract coordinates `(row, col)` and values `z` of valid (non-NaN) points.
2. Transform coordinates into polynomial features using `sklearn.preprocessing.PolynomialFeatures(degree=poly_degree)`.
   - Input features: `[row, col]` → transformed: `[1, row, col, row², row·col, col², row³, ...]`
3. Fit a regression model using `sklearn.linear_model.Ridge(alpha=ridge_alpha)`.
   - `X_train`: polynomial features of valid-value coordinates
   - `y_train`: valid values
4. Predict values at NaN coordinates → replace NaN with predictions.
5. **Valid values are preserved as-is** (interpolation is applied only to NaN positions).

**Implementation Notes:**
- For large data, normalize coordinates to [0, 1] for numerical stability:
  ```python
  row_norm = row / (H - 1)
  col_norm = col / (W - 1)
  ```
- Log a warning if valid values constitute less than 10% of total data.

### 4.5 Step 5 — Gaussian Smoothing

```python
def smooth_gaussian(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
```

**Purpose:** Remove residual micro-discontinuities after interpolation and produce a smooth surface suitable for trend visualization.

**Implementation:**
- Use `scipy.ndimage.gaussian_filter(data, sigma=sigma)`.
- Input data must be NaN-free (all values filled in Step 4).

**Saving:**
- Save result as `{original_filename}_preprocessed.txt`.
- Output location: `{subfolder}/interpolated/` directory (created if it does not exist).
- Format: tab-delimited, 4 decimal places.
  ```python
  np.savetxt(output_path, data, delimiter='\t', fmt='%.4f')
  ```

### 4.6 Step 6 — Global Min/Max Calculation

```python
def compute_global_minmax(root_dir: str) -> Tuple[float, float]:
```

**Purpose:** Compute global min and max across all preprocessed files in all subfolders to establish a unified image scaling baseline.

**Algorithm:**
1. Scan all `*/interpolated/*_preprocessed.txt` files under `root_dir`.
2. Load each file and extract its min and max.
3. Return the overall global min and global max.

### 4.7 Step 7 — Min-Max Scaling & Grayscale Image Generation

```python
def generate_grayscale_image(
    data: np.ndarray,
    global_min: float,
    global_max: float,
    output_path: str
) -> None:
```

**Algorithm:**
1. Min-Max Scaling: `scaled = (data - global_min) / (global_max - global_min)` → range [0, 1].
2. 8-bit conversion: `img = (scaled * 255).astype(np.uint8)`.
3. Save using `matplotlib.pyplot.imsave(output_path, img, cmap='gray')` or `PIL.Image.fromarray(img, mode='L').save(output_path)`.
4. Output location: `{subfolder}/images/` directory (created if it does not exist).
5. Filename: `{original_filename}_preprocessed.png`.

---

## 5. Main Orchestration

```python
def main(config: dict) -> None:
```

### Input Mode Branching

```
if config.input_file:
    → Single-file mode
elif config.root_dir:
    → Batch (directory) mode
else:
    → raise error
```

### Single-File Mode

When processing a single file, output folders are created relative to the file's **parent directory**.

```
filepath = config.input_file
parent_dir = dirname(filepath)

1. data = read_elevation(filepath)
2. data = downsample_median(data, factor)
3. data = detect_and_remove_outliers(data, grid_size, z_threshold)
4. data = interpolate_surface(data, poly_degree, ridge_alpha)
5. data = smooth_gaussian(data, sigma)
→ save to {parent_dir}/interpolated/{name}_preprocessed.txt

# Single file: the file itself defines the global range
global_min, global_max = data.min(), data.max()
generate_grayscale_image(data, global_min, global_max, output_path)
→ save to {parent_dir}/images/{name}_preprocessed.png
```

### Batch (Directory) Mode

#### Phase 1: Per-file Preprocessing (Steps 1–5)

```
for each subfolder in root_dir:
    for each .txt file in subfolder (excluding interpolated/, images/):
        if filename (without .txt) ends with any of exclude_suffixes → skip
        1. data = read_elevation(filepath)
        2. data = downsample_median(data, factor)
        3. data = detect_and_remove_outliers(data, grid_size, z_threshold)
        4. data = interpolate_surface(data, poly_degree, ridge_alpha)
        5. data = smooth_gaussian(data, sigma)
        → save to {subfolder}/interpolated/{name}_preprocessed.txt
```

#### Phase 2: Global Scaling & Image Generation (Steps 6–7)

```
global_min, global_max = compute_global_minmax(root_dir)

for each subfolder in root_dir:
    for each *_preprocessed.txt in subfolder/interpolated/:
        data = load preprocessed file
        generate_grayscale_image(data, global_min, global_max, output_path)
        → save to {subfolder}/images/{name}_preprocessed.png
```

---

## 6. Dependencies

| Package | Version | Usage |
|---|---|---|
| `numpy` | ≥ 1.24 | Array operations, I/O |
| `scipy` | ≥ 1.10 | `gaussian_filter` |
| `scikit-learn` | ≥ 1.2 | `PolynomialFeatures`, `Ridge` |
| `matplotlib` | ≥ 3.7 | Image saving (optional) |
| `Pillow` | ≥ 9.0 | Image saving (optional) |

```
pip install numpy scipy scikit-learn matplotlib Pillow
```

---

## 7. Module Structure

```
pcb_elevation_preprocessor/
├── main.py                 # CLI entry point & orchestration
├── config.py               # Configuration dataclass & defaults
├── io_utils.py             # read_elevation(), save_preprocessed(), file discovery
├── preprocessing.py        # downsample, outlier detection, interpolation, smoothing
├── imaging.py              # global minmax, scaling, grayscale image generation
└── README.md
```

---

## 8. CLI Interface

### Mode 1: Directory (Batch) — Process all subfolders

```bash
python main.py \
    --root-dir /path/to/data \
    --downsample-factor 4 \
    --z-threshold 3.0 \
    --poly-degree 3 \
    --ridge-alpha 0.1 \
    --gaussian-sigma 1.0
```

### Mode 2: Single File — Process a single txt file

```bash
python main.py \
    --input-file /path/to/single_sample.txt \
    --downsample-factor 4
```

- `--root-dir` and `--input-file` are **mutually exclusive** (specifying both raises an error).
- Specifying neither also raises an error.

All other arguments are optional and override the config defaults.

---

## 9. Logging

- Uses the `logging` module (level: INFO).
- Key log points:
  - File load: output shape.
  - Downsampling: shape before and after.
  - Outlier detection: per-region count, total removed count.
  - Interpolation: valid value ratio (%), training completion.
  - Smoothing: completion.
  - Global min/max values.
  - Image save: completion.

---

## 10. Error Handling

| Scenario | Action |
|---|---|
| File is not tab-delimited | Log error, skip the file |
| Entire data is NaN (no valid values) | Log warning, skip the file |
| Valid values < 5% of total during interpolation | Log warning (potential quality degradation) |
| No .txt files in subfolder | Log info, skip |
| `global_min == global_max` | Scaling not possible → generate all images with mid-value (128) |

---

## 11. Testing Strategy

- **Unit Tests:** Validate each function using small synthetic data (e.g., 16×16).
  - `read_elevation`: Verify 9999.0 → NaN conversion.
  - `downsample_median`: Verify median results against known values.
  - `detect_and_remove_outliers`: Insert artificial outliers, confirm detection.
  - `interpolate_surface`: Verify reasonable values are generated at NaN positions.
  - `smooth_gaussian`: Verify output shape is unchanged, value range is reasonable.
- **Integration Test:** Run the full pipeline on realistic-size dummy data → verify existence of result files and images.
