# PCB Warpage — Project Guide

## Overview

Generative modeling project for PCB (Printed Circuit Board) warpage/elevation pattern generation. Supports four conditional generative architectures — **CVAE**, **DDPM**, **LDM**, **LFM** — in a unified, config-driven framework. Uses a **leave-one-out** validation protocol across 10 PCB designs (A–J).

## Architecture

```
pcb_warpage/
├── models/                         # Model implementations
│   ├── __init__.py                 # Factory: build_model(config)
│   ├── cvae.py                     # Conditional VAE (3 fusion methods)
│   ├── ddpm.py                     # Pixel-space diffusion
│   ├── ldm.py                      # Latent diffusion (frozen CVAE + denoiser)
│   ├── lfm.py                      # Latent flow matching (frozen CVAE + velocity net)
│   ├── design_encoder.py           # CNN + handcrafted features → condition vector
│   ├── elevation_encoder.py        # CNN → (μ, logvar)
│   ├── decoder.py                  # FiLM-conditioned upsampling decoder
│   ├── unet.py                     # U-Net for DDPM noise prediction
│   ├── latent_denoiser.py          # MLP denoiser / velocity net (LDM & LFM)
│   └── ddpm_condition_encoder.py   # Multi-scale CNN for DDPM conditioning
│
├── utils/
│   ├── load_config.py              # Key-value config parser
│   ├── dataset.py                  # PCBWarpageDataset with D4 augmentation
│   ├── handcrafted_features.py     # 24-dim design feature extractor
│   ├── losses.py                   # CVAE loss, KL annealing, free-bits, spectral loss
│   └── ema.py                      # Exponential Moving Average
│
├── data_generation/                # Synthetic data generation (designs A–J)
├── documents/                      # Architecture specs (Korean)
├── preprocess/                     # Preprocessing utilities and specs
├── data/                           # Design images + elevation samples per design
│   ├── design/                     #   design_{A..J}.png
│   └── elevation/design_*/images/  #   Per-design elevation PNGs
├── outputs/                        # Checkpoints, logs, visualizations
│
├── train.py                        # Main training script (all 4 models)
├── evaluate.py                     # Leave-one-out evaluation
├── sample.py                       # Inference / generation
├── analyze_features.py             # Feature importance ranking
├── config.txt                      # CVAE config
├── config_ddpm.txt                 # DDPM config
├── config_ldm.txt                  # LDM config
└── config_lfm.txt                  # LFM config
```

## Model Relationships

- **CVAE** — standalone baseline. Trains encoder, decoder, and fusion from scratch.
- **DDPM** — standalone pixel-space diffusion with its own U-Net and condition encoder.
- **LDM** — two-stage: freezes a pretrained CVAE, trains only a `LatentDenoiser` MLP in the CVAE's 64-dim latent space.
- **LFM** — two-stage: same frozen CVAE, trains a `VelocityNet` (same architecture as `LatentDenoiser`) for ODE-based flow matching.

All four models expose a unified interface:
```python
model.forward(elevation, design, hand_features)       # training
model.sample(design, hand_features, num_samples, temperature)  # inference
```

## Coding Conventions

### Naming
- **Variables / functions**: `snake_case` — `z_dim`, `design_encoder`, `_parse_value()`
- **Classes**: `PascalCase` — `CVAE`, `DesignEncoder`, `LatentDenoiser`
- **Constants**: `UPPER_SNAKE_CASE` — `HAND_FEATURE_DIM = 24`
- **Config keys**: `lower_snake_case` — `model_type`, `ldm_hidden_dim`
- **Private helpers**: prefix with `_` — `_conv_block()`, `_parse_value()`

### Style
- Python 3.10+ type hints (`tuple[Tensor, Tensor]`, `X | None`)
- Triple-quoted docstrings with Args/Returns sections
- ASCII architecture diagrams in module-level docstrings
- Inline comments for non-obvious logic; `# ---` section separators
- Import order: stdlib → third-party (torch, PIL, numpy) → local (utils, models)

### Patterns
- **Factory pattern**: `build_model(config)` in `models/__init__.py`
- **Config-driven**: all hyperparameters from `.txt` config files, never hardcoded
- **`@torch.no_grad()`** on all inference/sampling methods
- **EMA** applied at inference for DDPM/LDM/LFM
- **AMP** (mixed precision) for training; gradient clipping at max norm 5.0
- **Cosine annealing** LR scheduler

## Config Format

Plain-text key-value files (`config*.txt`). Sections marked with `%`, inline comments with `#`:

```
%   [Section]
key_name    value    # comment
```

Supported types: int, float, bool (`True`/`False`), string, comma-separated lists.

Key groups: model selection, paths, dataset, image size, network dimensions, training hyperparams, augmentation, inference settings, physical scaling.

## Data Pipeline

1. **Load**: grayscale design + elevation PNGs (256×256), extract 24 handcrafted features at original resolution
2. **Augment** (train only): D4 symmetry (8 orientations), ±5° rotation, brightness/contrast jitter, smooth low-freq elevation noise. Handcrafted features are **recomputed** after augmentation.
3. **Forward**: model-specific (see model docstrings for details)
4. **Loss**: CVAE uses MSE + cyclical β·KL + optional spectral FFT loss; DDPM/LDM/LFM use simple MSE on noise or velocity

## Evaluation

Leave-one-out protocol: train on 9 designs, evaluate on the held-out design.

Metrics:
- **Reconstruction MSE** (CVAE only) — deterministic path using μ
- **Sample Diversity** — per-pixel variance across K generated samples
- **MMD** — Maximum Mean Discrepancy between real and generated distributions
- **Active KL dims** (CVAE only) — dims with mean KL > 0.1 nats

## Commands

```bash
# Training
python train.py --config config.txt --val_fold 0

# Evaluation (all folds or single fold)
python evaluate.py --fold 0 --k 50

# Inference
python sample.py --design data/design/design_A.png --num_samples 10
python sample.py --design-dir data/design/ --denormalize

# Feature analysis
python analyze_features.py
```

## Checkpoint Format

```python
{
    'epoch': int,
    'model_type': 'cvae' | 'ddpm' | 'ldm' | 'lfm',
    'model_state': OrderedDict,
    'optimizer_state': OrderedDict,
    'config': dict,
    'ema_state_dict': dict,       # DDPM/LDM/LFM only
    'val_loss': float,
    'cvae_checkpoint': str,       # LDM/LFM only
}
```

## Key Dimensions

| Component | Default Dim | Notes |
|-----------|-------------|-------|
| Image size | 256×256 | Configurable to 128 |
| Condition `c` | 64 | Design encoder output |
| Stochastic latent `z₁` | 64 | Elevation encoder output |
| Handcrafted features | 24 | Selectable subset via config |
| Denoiser/velocity hidden | 512 | 8 AdaLN-ResBlocks |
| DDPM timesteps | 1000 | Cosine schedule |
| LDM DDIM steps | 50 | Inference |
| LFM ODE steps | 30 | Euler integration |

## Dependencies

- `torch >= 2.0`, `torchvision >= 0.15` — core ML framework (pure PyTorch, no external trainers)
- `numpy`, `scipy` — numerics and signal processing
- `Pillow` — image I/O
- `matplotlib` — visualization and colormaps
- Python 3.12+
