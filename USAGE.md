# CVAE PCB Warpage — Usage Guide

---

## Project Structure

```
pcb_warpage/
├── config.txt                          # All hyperparameters
├── train.py                            # Training loop
├── evaluate.py                         # Leave-one-out evaluation
├── sample.py                           # Inference / generation
├── analyze_features.py                 # Feature importance ranking & selection
├── data_generation/
│   ├── generate_design.py              # Synthetic design images (10 variants, A–J)
│   ├── generate_elevation.py           # Synthetic elevation images (300 per design)
│   └── visualize_samples.py            # Sanity-check plots
├── models/
│   ├── design_encoder.py               # CNN + handcrafted features → c (deterministic)
│   ├── elevation_encoder.py            # CNN → μ, logvar, z1 (stochastic)
│   ├── decoder.py                      # FiLM-conditioned upsampler
│   └── cvae.py                         # Full CVAE (Concat / FiLM / CrossAttn)
└── utils/
    ├── load_config.py                  # config.txt parser
    ├── handcrafted_features.py         # 22-dim design feature extractor
    ├── losses.py                       # MSE recon + KL + cyclical β annealing
    └── dataset.py                      # PCBWarpageDataset + DataLoader factory
```

---

## Prerequisites

```bash
pip install -r requirements.txt
```

---

## Option A — Using Synthetic Data (Development)

Use this path when real PCB data is unavailable.

### Step 1 — Generate design images

```bash
python -m data_generation.generate_design
```

Creates `data/design/design_{A,...,J}.png` — 10 grayscale layout variants.
Each image contains outline-only polygons arranged in a symmetric grid
that is auto-scaled to fill the canvas:

| Design | Shape | Grid | Copies |
|--------|-------|------|--------|
| A | Complex DIP outline — pin-slot notches on all four sides | 2 × 2 | 4 |
| B | Winding staircase — concave stepped outline | 3 × 2 | 6 |
| C | Notched cross — 2 rectangular slots cut into each arm tip | 2 × 2 | 4 |
| D | Notched chamfered octagon — inward notch on each flat edge | 2 × 3 | 6 |
| E | Ascending staircase band | 2 × 2 | 4 |
| F | T-shape with pin-slot notches | 2 × 2 | 4 |
| G | Hourglass with notched ends | 2 × 2 | 4 |
| H | Three-tooth comb | 2 × 2 | 4 |
| I | I-beam with notched flanges | 2 × 2 | 4 |
| J | U-channel with wall notches | 2 × 2 | 4 |

### Step 2 — Generate elevation images

```bash
python -m data_generation.generate_elevation
```

Creates 300 elevation samples per design under `data/elevation/design_{A,...,J}/`.

Each sample is generated as:
```
elevation = pattern_surface(random_type) × amplitude
          + density_map × 0.12
          + low_freq_noise()
          + random_tilt()
elevation = normalise(elevation) → [0, 1]
```

The pattern type is drawn uniformly at random for every sample:

| Pattern type | Description |
|---|---|
| `center_bump` | Gaussian peak near the centre — centre is highest |
| `center_bowl` | Inverted Gaussian — centre is lowest, edges high |
| `corner_single` | One random corner elevated **or** depressed (50 / 50) |
| `corner_diagonal` | Two diagonal corners high **or** low (saddle) |
| `corner_adjacent` | Two edge-adjacent corners high **or** low (ramp) |
| `corner_all` | All four corners high **or** low |

Per-design warp amplitude and tilt scale are individually configured
in `generate_elevation.py` to reflect realistic material differences.

### Step 3 — Visualise (optional sanity check)

```bash
python -m data_generation.visualize_samples
```

Plots a grid of design images alongside random elevation samples, and a
per-pixel variance map showing within-design diversity.

### Step 4 — Train

```bash
python train.py
```

No changes to `config.txt` needed for synthetic data.

---

## Option B — Using Real Data

### Your data can be in any layout. Configure it in `config.txt`.

#### Example layout

```
D:/path_to_elevation/
├── A.png               ← design image for A
├── B.png               ← design image for B
├── C.png               ← design image for C
├── A/
│   └── images/         ← elevation PNGs for A
├── B/
│   └── images/         ← elevation PNGs for B
└── C/
    └── images/         ← elevation PNGs for C
```

#### Corresponding `config.txt` (Dataset section)

```
%   Dataset
dataset_dir             D:/path_to_elevation
design_names            A, B, C
design_image_dir        D:/path_to_elevation
elevation_base_dir      D:/path_to_elevation
elevation_subdir        images
num_designs             3
val_fold                0     # 0=A held-out, 1=B held-out, 2=C held-out
```

#### How the paths are resolved

| Config key | Role | Resolved path |
|---|---|---|
| `design_names` | Comma-separated design labels | `A`, `B`, `C` |
| `design_image_dir` | Folder containing `{name}.png` | `D:/path_to_elevation/A.png` |
| `elevation_base_dir` | Folder containing `{name}/` subfolders | `D:/path_to_elevation/A/` |
| `elevation_subdir` | Subfolder inside each `{name}/` dir | `D:/path_to_elevation/A/images/` |

If your elevation images are **directly** inside `A/` (no `images/` subfolder), omit
`elevation_subdir` or leave it blank.

#### What the image files must be

| Type | Format | Expected pixel values |
|---|---|---|
| Design image | Grayscale PNG | White background (255), black lines (0) |
| Elevation image | Grayscale PNG | Smooth gradient, full 0–255 range |

Both are automatically resized to `image_size × image_size` (default 256×256).

### Step 1 — Edit `config.txt` as shown above, then train

```bash
python train.py
```

For a different held-out fold:

```bash
python train.py --val_fold 1    # holds out design B
python train.py --val_fold 2    # holds out design C
```

---

## Training

```
python train.py [--config config.txt] [--val_fold N]
```

Key config parameters:

| Key | Default | Description |
|---|---|---|
| `training_epochs` | 50 | Total epochs |
| `early_stop_threshold` | 0.001 | Halt when val recon < this; 0 = disabled |
| `batch_size` | 32 | Mini-batch size |
| `learning_rate` | 0.0001 | Adam LR |
| `beta_max` | 0.5 | Max KL weight β |
| `beta_cycles` | 4 | Cyclical annealing cycles |
| `fusion_method` | `film` | `concat` / `film` / `cross_attention` |
| `val_fold` | 0 | Which design to hold out (0-indexed) |

Checkpoints are saved to `modelpath` whenever validation reconstruction loss improves.
Logs are written to `log_file_dir`.

### Early stopping

Training halts automatically when validation reconstruction loss drops below
`early_stop_threshold`. Set to `0` to disable.

### Fusion method comparison (from spec)

| Step | Fusion | Purpose |
|---|---|---|
| 1 | `concat` | Establish baseline |
| 2 | `film` | Measure conditioning improvement |
| 3 | `cross_attention` | After data scale-up (10+ designs) |

---

## Evaluation

Leave-one-out evaluation over all designs:

```bash
python evaluate.py
```

Single fold only:

```bash
python evaluate.py --fold 0
python evaluate.py --fold 1 --k 20    # override number of generated samples
```

Metrics reported per fold:

| Metric | Description |
|---|---|
| Reconstruction MSE | MSE of deterministic reconstruction (μ path, no sampling) |
| Sample Diversity | Mean per-pixel variance across K generated samples |
| MMD | Maximum Mean Discrepancy between generated and real distributions |

---

## Sampling (Inference)

Generate K elevation samples for a given design image:

```bash
python sample.py --design data/design/design_A.png
python sample.py --design data/design/design_A.png --k 16
python sample.py --design data/design/design_A.png --k 16 --temperature 1.2
python sample.py --design data/design/design_A.png --k 16 --save outputs/samples_A.png
python sample.py --design D:/path_to_elevation/A.png --k 10
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--design` | (required) | Path to design PNG |
| `--k` | from config | Number of samples to generate |
| `--temperature` | 1.0 | Scale factor on z1 prior; >1 = more diverse, <1 = less |
| `--checkpoint` | from config | Override model checkpoint path |
| `--save` | (show plot) | Save figure to file instead of displaying |
| `--config` | `config.txt` | Config file path |

---

## Feature Selection

The handcrafted feature extractor computes 22 features from each design image.
When data is limited, using all 22 may introduce noise. The selection tool
identifies the most informative subset.

### Analyse and rank features

```bash
python analyze_features.py --top 10
```

This prints:
- Feature values for each design
- Elevation statistics (if elevation data exists)
- A ranked list by importance (variance × relevance × non-redundancy)
- The recommended top-10 feature indices

### Apply selection

```bash
python analyze_features.py --top 10 --write
```

Writes `selected_features  0,7,11,13,14,15,18,19,20,21` (example) to `config.txt`.
The design encoder reads this key and uses only those features during training and inference.

To revert to all 22 features, remove `selected_features` from `config.txt`.

### Options

| Flag | Default | Description |
|---|---|---|
| `--top` | 22 (all) | Number of features to select |
| `--corr_threshold` | 0.85 | Max |correlation| allowed between selected features |
| `--write` | off | Write selection to config.txt |
| `--design_dir` | `./data/design` | Design image directory |
| `--elevation_dir` | `./data/elevation` | Elevation base directory |
| `--config` | `./config.txt` | Config file to update |

---

## Typical Full Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2a. Synthetic data — generate
python -m data_generation.generate_design
python -m data_generation.generate_elevation
python -m data_generation.visualize_samples   # optional

# 2b. Real data — just edit config.txt (no generation needed)

# 3. (Optional) Analyse features and select top-k
python analyze_features.py --top 10 --write

# 4. Train (fold 0 = hold out first design)
python train.py

# 5. Evaluate all folds
python evaluate.py

# 6. Generate samples for a design of interest
python sample.py --design data/design/design_C.png --k 16 --save outputs/design_C_samples.png
```

---

## Config Reference

Full list of `config.txt` keys:

```
%   Model / Mode
model               CVAE_PCB
mode                Train       # Train / Inference
gpu_ids             0           # -1 for CPU

%   Paths
log_file_dir        ./outputs/train.log
modelpath           ./outputs/cvae_pcb.pth

%   Dataset
dataset_dir         ./data
design_names        A, B, C             # optional: overrides default design_A…J
design_image_dir    ./data/design       # optional: folder with {name}.png files
elevation_base_dir  ./data/elevation    # optional: folder with {name}/ subfolders
elevation_subdir                        # optional: subfolder inside {name}/
num_designs         10

%   Image
image_size          256

%   Network
z_dim               64      # stochastic latent dim (z1)
c_dim               64      # condition vector dim
c_cnn_dim           64      # design CNN branch output
c_hand_dim          32      # handcrafted feature branch output
fusion_method       film    # concat / film / cross_attention
selected_features           # optional: comma-separated feature indices, e.g. 0,7,11,13

%   Training
training_epochs     50
early_stop_threshold 0.001  # halt when val recon < this; 0 = disabled
batch_size          32
learning_rate       0.0001
num_workers         4
weight_decay        0.0001

%   KL annealing
beta_start          0.0
beta_max            0.5
beta_cycles         4

%   Augmentation
use_design_aug      True
use_elevation_aug   True

%   Validation
val_fold            0       # index of held-out design (0-indexed)

%   Inference
num_gen_samples     10      # K samples per design at inference
```
