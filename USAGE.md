# CVAE PCB Warpage — Usage Guide

---

## Project Structure

```
pcb_warpage/
├── config.txt                          # All hyperparameters
├── train.py                            # Training loop
├── evaluate.py                         # Leave-one-out evaluation
├── sample.py                           # Inference / generation
├── data_generation/
│   ├── generate_design.py              # Synthetic design images (4 variants)
│   ├── generate_elevation.py           # Synthetic elevation images
│   └── visualize_samples.py            # Sanity-check plots
├── models/
│   ├── design_encoder.py               # CNN + handcrafted features → c (deterministic)
│   ├── elevation_encoder.py            # CNN → μ, logvar, z1 (stochastic)
│   ├── decoder.py                      # FiLM-conditioned upsampler
│   └── cvae.py                         # Full CVAE (Concat / FiLM / CrossAttn)
└── utils/
    ├── load_config.py                  # config.txt parser
    ├── handcrafted_features.py         # 10-dim design feature extractor
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

Creates `data/design/design_{A,B,C,D}.png` — 4 grayscale layout variants.
Each image contains outline-only polygons (no fill) arranged in a symmetric
grid that is auto-scaled to fill the canvas:

| Design | Shape | Grid | Copies |
|--------|-------|------|--------|
| A | Complex DIP outline — pin-slot notches on all four sides | 2 × 2 | 4 |
| B | Winding staircase — concave stepped outline | 3 × 2 | 6 |
| C | Notched cross — 2 rectangular slots cut into each arm tip | 2 × 2 | 4 |
| D | Notched chamfered octagon — inward notch on each flat edge | 2 × 3 | 6 |

### Step 2 — Generate elevation images

```bash
python -m data_generation.generate_elevation
```

Creates 300 elevation samples per design under `data/elevation/design_{A,B,C,D}/`.

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

The density-map term (`× 0.12`) preserves a weak design-specific bias so the
CVAE can still learn design conditioning.

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
| `training_epochs` | 200 | Total epochs |
| `batch_size` | 32 | Mini-batch size |
| `learning_rate` | 0.0001 | Adam LR |
| `beta_max` | 4.0 | Max KL weight β |
| `beta_cycles` | 4 | Cyclical annealing cycles |
| `fusion_method` | `film` | `concat` / `film` / `cross_attention` |
| `val_fold` | 0 | Which design to hold out (0-indexed) |

Checkpoints are saved to `modelpath` whenever validation reconstruction loss improves.
Logs are written to `log_file_dir`.

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
python sample.py --design data/design/design_A.png --k 16 --save outputs/samples_A.png
python sample.py --design D:/path_to_elevation/A.png --k 10
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--design` | (required) | Path to design PNG |
| `--k` | from config | Number of samples to generate |
| `--checkpoint` | from config | Override model checkpoint path |
| `--save` | (show plot) | Save figure to file instead of displaying |
| `--config` | `config.txt` | Config file path |

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

# 3. Train (fold 0 = hold out first design)
python train.py

# 4. Evaluate all folds
python evaluate.py

# 5. Generate samples for a design of interest
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
design_names        A, B, C             # optional: overrides default design_A…D
design_image_dir    ./data/design       # optional: folder with {name}.png files
elevation_base_dir  ./data/elevation    # optional: folder with {name}/ subfolders
elevation_subdir                        # optional: subfolder inside {name}/
num_designs         4

%   Image
image_size          256

%   Network
z_dim               64      # stochastic latent dim (z1)
c_dim               64      # condition vector dim
c_cnn_dim           64      # design CNN branch output
c_hand_dim          16      # handcrafted feature branch output
fusion_method       film    # concat / film / cross_attention

%   Training
training_epochs     200
batch_size          32
learning_rate       0.0001
num_workers         4
weight_decay        0.0001

%   KL annealing
beta_start          0.0
beta_max            4.0
beta_cycles         4

%   Augmentation
use_design_aug      True
use_elevation_aug   True

%   Validation
val_fold            0       # index of held-out design (0-indexed)

%   Inference
num_gen_samples     10      # K samples per design at inference
```
