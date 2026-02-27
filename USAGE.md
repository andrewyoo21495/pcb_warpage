# PCB Warpage — Usage Guide

---

## Project Structure

```
pcb_warpage/
├── config.txt                          # All hyperparameters (model_type, CVAE, DDPM)
├── train.py                            # Training loop (CVAE or DDPM)
├── evaluate.py                         # Leave-one-out evaluation
├── sample.py                           # Inference / generation
├── analyze_features.py                 # Feature importance ranking & selection
├── data_generation/
│   ├── generate_design.py              # Synthetic design images (10 variants, A–J)
│   ├── generate_elevation.py           # Synthetic elevation images (per design)
│   └── visualize_samples.py            # Sanity-check plots
├── models/
│   ├── __init__.py                     # build_model() factory (cvae / ddpm)
│   ├── cvae.py                         # Full CVAE (Concat / FiLM / CrossAttn)
│   ├── design_encoder.py              # CNN + handcrafted features → c (deterministic)
│   ├── elevation_encoder.py           # CNN → μ, logvar, z1 (stochastic)
│   ├── decoder.py                     # FiLM-conditioned upsampler
│   ├── ddpm.py                        # Conditional DDPM (cosine schedule + DDIM)
│   ├── ddpm_condition_encoder.py      # Multi-scale CNN → spatial feats + global cond
│   └── unet.py                        # U-Net noise predictor with AdaGN
└── utils/
    ├── load_config.py                  # config.txt parser
    ├── handcrafted_features.py         # 24-dim design feature extractor
    ├── losses.py                       # MSE recon + KL + cyclical β annealing (CVAE)
    ├── dataset.py                      # PCBWarpageDataset + DataLoader factory
    └── ema.py                          # Exponential Moving Average (DDPM)
```

---

## Model Selection

This project supports two generative model architectures:

| Model | Config value | Description |
|---|---|---|
| **CVAE** | `model_type  cvae` | Conditional VAE with 3 fusion methods (default) |
| **DDPM** | `model_type  ddpm` | Conditional Denoising Diffusion Probabilistic Model |

Set the model type in `config.txt`:
```
model_type  cvae    # or: ddpm
```

Both models share the same data pipeline, config system, evaluation metrics, and
sampling interface. The `model_type` key determines which model is built, trained,
and evaluated.

### Architecture Comparison

| | CVAE | DDPM |
|---|---|---|
| **Parameters** | ~4.2M | ~14.3M |
| **Condition encoder** | CNN → 64-dim vector | Multi-scale CNN → spatial maps + 256-dim vector |
| **Generation method** | Single forward pass (decoder) | Iterative reverse diffusion (DDIM, 50 steps) |
| **Stochasticity** | Latent z1 ~ N(0,I) | Pure noise → iterative denoising |
| **Training loss** | MSE recon + β·KL | MSE noise prediction |
| **Inference speed** | Fast (single pass) | Slower (50 DDIM steps) |
| **Strengths** | Fast inference, explicit latent space | Superior spatial detail, better mode coverage |

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

Creates elevation samples per design under `data/elevation/design_{A,...,J}/images/`.

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

No changes to `config.txt` needed for synthetic data (default: `model_type cvae`).

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

The training loop automatically adapts to the selected `model_type`.

### CVAE Training

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

- Uses Adam optimizer with cosine annealing LR schedule
- Cyclical KL annealing: β ramps from 0 → β_max over each cycle
- Checkpoint saved when validation reconstruction loss improves
- Logs written to `log_file_dir`

### DDPM Training

| Key | Default | Description |
|---|---|---|
| `training_epochs` | 2000 (recommended) | Total epochs (DDPM needs more) |
| `early_stop_threshold` | 0 | Halt when val noise-pred loss < this; 0 = disabled |
| `batch_size` | 16–32 | Mini-batch size |
| `learning_rate` | 0.0001 | AdamW LR |
| `ddpm_T` | 1000 | Diffusion timesteps |
| `ddpm_base_ch` | 64 | U-Net base channels |
| `ddpm_cond_dim` | 256 | Global condition vector dimension |
| `ddpm_dropout` | 0.15 | Dropout in U-Net ResBlocks |
| `ema_decay` | 0.9999 | EMA decay (critical for generation quality) |

- Uses AdamW optimizer with cosine annealing LR schedule
- EMA (Exponential Moving Average) updated after every optimizer step
- Training loss: MSE between predicted and actual noise
- Checkpoint includes both model weights and EMA weights
- At inference, EMA weights are used (smoother, higher quality outputs)

### CVAE Fusion method comparison (from spec)

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

The evaluation script auto-detects the model type from the checkpoint file.

Metrics reported per fold:

| Metric | CVAE | DDPM | Description |
|---|---|---|---|
| Reconstruction MSE | Yes | N/A | MSE of deterministic reconstruction (μ path) |
| Sample Diversity | Yes | Yes | Mean per-pixel variance across K generated samples |
| MMD | Yes | Yes | Maximum Mean Discrepancy between generated and real distributions |

For DDPM, reconstruction MSE is not applicable since diffusion models don't have
a deterministic reconstruction path.

---

## Sampling (Inference)

Generate K elevation samples for a given design image:

```bash
python sample.py --design data/design/design_A.png
python sample.py --design data/design/design_A.png --k 16
python sample.py --design data/design/design_A.png --k 16 --temperature 1.2
python sample.py --design data/design/design_A.png --k 16 --save outputs/samples_A/
python sample.py --design D:/path_to_elevation/A.png --k 10
python sample.py --design data/design/design_A.png --k 10 --denormalize
```

The script auto-detects the model type from the checkpoint.

### Batch mode — process all designs in a folder

Instead of generating samples for a single design, you can provide a folder containing multiple design PNGs. The script iterates through all `.png` files and generates K samples for each:

```bash
python sample.py --design-dir data/design/ --k 16 --save outputs/samples/
```

Output structure — a separate subfolder is created for each design:

```
outputs/samples/
├── design_A/
│   ├── elevation_0001.png
│   ├── ...
│   └── elevation_0016.png
├── design_B/
│   ├── elevation_0001.png
│   └── ...
└── design_J/
    └── ...
```

All other flags (`--temperature`, `--colormap`, `--save_txt`, `--denormalize`) work with batch mode.

### Elevation range histograms for generated samples

In batch mode (`--design-dir`), after generating all samples, the script automatically plots a per-design elevation range distribution histogram and saves them to:

```
outputs/distribution/generated/
├── dist_design_A.png
├── dist_design_B.png
└── ...
```

Each histogram shows the distribution of per-sample elevation ranges (max - min) with a mean marker line. This lets you verify that the generated samples have realistic and consistent range distributions across designs.

### Temperature behavior

| Model | `--temperature` effect |
|---|---|
| CVAE | Scales z1 prior noise; >1 = more diverse, <1 = less |
| DDPM | Scales DDIM eta (stochasticity); >1 = more diverse, <1 = more deterministic |

### Denormalization

The `--denormalize` flag applies the inverse min-max transform to convert generated pixel
values back to their original physical units (e.g., mm).  It reads `elevation_min` and
`elevation_max` from `config.txt` and writes a `.txt` file alongside each PNG:

```
Forward:  pixel ∈ [0, 1]  ←  (value − elev_min) / (elev_max − elev_min)
Inverse:  value           =   pixel × (elev_max − elev_min) + elev_min
```

Set the two keys in `config.txt` before using this flag:
```
elevation_min   -0.5    # your actual physical minimum (e.g. mm)
elevation_max    2.3    # your actual physical maximum
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--design` | — | Path to a single design PNG (use this **or** `--design-dir`) |
| `--design-dir` | — | Path to a folder of design PNGs for batch processing |
| `--k` | from config | Number of samples to generate |
| `--temperature` | 1.0 | Diversity control (see table above) |
| `--denormalize` | off | Save `.txt` files with physical values (requires `elevation_min`/`max` in config) |
| `--colormap` | — | Apply a matplotlib colormap (e.g. `jet`) to saved PNGs |
| `--save_txt` | off | Save as tab-delimited `.txt` files (reverse of preprocessing) |
| `--metadata` | auto-detect | Path to `scaling_metadata.json` (for `--save_txt`) |
| `--checkpoint` | from config | Override model checkpoint path |
| `--save` | `outputs/samples` | Directory to save individual PNG files (or base dir for batch mode) |
| `--config` | `config.txt` | Config file path |

---

## Feature Selection

The handcrafted feature extractor computes 24 features from each design image.
When data is limited, using all 24 may introduce noise. The selection tool
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

To revert to all 24 features, remove `selected_features` from `config.txt`.

### Options

| Flag | Default | Description |
|---|---|---|
| `--top` | 24 (all) | Number of features to select |
| `--corr_threshold` | 0.85 | Max \|correlation\| allowed between selected features |
| `--write` | off | Write selection to config.txt |
| `--design_dir` | from config (`dataset_dir/design`) | Override design image directory |
| `--elevation_dir` | from config (`dataset_dir/elevation`) | Override elevation base directory |
| `--config` | `./config.txt` | Config file to read and optionally update |

---

## Typical Full Workflow

### CVAE Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2a. Synthetic data — generate
python -m data_generation.generate_design
python -m data_generation.generate_elevation
python -m data_generation.visualize_samples   # optional

# 2b. Real data — just edit config.txt (no generation needed)

# 3. Ensure model_type is set to cvae in config.txt
#    model_type  cvae

# 4. (Optional) Analyse features and select top-k
python analyze_features.py --top 10 --write

# 5. Train (fold 0 = hold out first design)
python train.py

# 6. Evaluate all folds
python evaluate.py

# 7. Generate samples for a design of interest
python sample.py --design data/design/design_C.png --k 16 --save outputs/design_C_samples/

# 7b. Or generate for ALL designs at once (creates subfolders per design)
python sample.py --design-dir data/design/ --k 16 --save outputs/samples/
# -> outputs/samples/design_A/, design_B/, ...
# -> outputs/distribution/generated/dist_design_A.png, ...
```

### DDPM Workflow

```bash
# 1–2. Same as CVAE (data preparation)

# 3. Set model_type to ddpm in config.txt
#    model_type  ddpm
#    training_epochs  2000      # DDPM needs more epochs
#    modelpath  ./outputs/ddpm_pcb.pth   # separate checkpoint path

# 4. Train
python train.py

# 5. Evaluate (auto-detects DDPM from checkpoint)
python evaluate.py

# 6. Generate samples (auto-detects DDPM from checkpoint)
python sample.py --design data/design/design_C.png --k 16 --save outputs/ddpm_samples_C/

# 6b. Or batch generate for all designs
python sample.py --design-dir data/design/ --k 16 --save outputs/ddpm_samples/
```

---

## Config Reference

Full list of `config.txt` keys:

```
%   Model selection
model_type          cvae        # cvae / ddpm
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
elevation_subdir    images              # optional: subfolder inside {name}/
num_designs         10

%   Image
image_size          256

%   CVAE network
z_dim               64      # stochastic latent dim (z1)
c_dim               64      # condition vector dim
c_cnn_dim           64      # design CNN branch output
c_hand_dim          32      # handcrafted feature branch output
fusion_method       film    # concat / film / cross_attention
selected_features           # optional: comma-separated feature indices, e.g. 0,7,11,13

%   Training (shared)
training_epochs     50      # CVAE: ~50–200; DDPM: ~2000
early_stop_threshold 0.001  # halt when val loss < this; 0 = disabled
batch_size          32
learning_rate       0.0001
num_workers         4
weight_decay        0.0001

%   KL annealing (CVAE only)
beta_start          0.0
beta_max            0.5
beta_cycles         4

%   DDPM parameters
ddpm_T              1000    # diffusion timesteps
ddpm_base_ch        64      # U-Net base channels
ddpm_cond_dim       256     # global condition vector dimension
ddpm_eta            0.7     # DDIM stochasticity: 0=deterministic, 1=full DDPM
ddpm_ddim_steps     50      # DDIM inference steps
ddpm_dropout        0.15    # dropout in U-Net ResBlocks
ema_decay           0.9999  # EMA decay for DDPM

%   Augmentation
use_design_aug      True
use_elevation_aug   True

%   Validation
val_fold            0       # index of held-out design (0-indexed)

%   Inference
num_gen_samples     10      # K samples per design at inference

%   Physical scaling  (for sample.py --denormalize)
elevation_min       0.0     # physical minimum value used in original min-max scaling
elevation_max       1.0     # physical maximum value used in original min-max scaling
```

---

## Checkpoint Format

Checkpoints are saved as PyTorch `.pth` files with model-type-specific contents:

### CVAE Checkpoint
```python
{
    'epoch':           int,
    'model_type':      'cvae',
    'model_state':     OrderedDict,  # model.state_dict()
    'optimizer_state':  OrderedDict,
    'val_loss':        float,
    'val_recon':       float,
    'config':          dict,
}
```

### DDPM Checkpoint
```python
{
    'epoch':           int,
    'model_type':      'ddpm',
    'model_state':     OrderedDict,  # model.state_dict() (non-EMA, for training resume)
    'ema_state_dict':  dict,         # EMA weights (used for inference)
    'optimizer_state':  OrderedDict,
    'val_loss':        float,
    'config':          dict,
}
```

When loading checkpoints for evaluation or sampling, the scripts auto-detect
`model_type` and use EMA weights for DDPM.
