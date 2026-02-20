# CVAE-Based PCB Warpage Distribution Generation System

---

## 1. Problem Definition

PCBs manufactured from the same design can exhibit different warpage patterns due to varying production conditions.

**Goal:** Given a PCB design image (C), model the conditional distribution p(X|C) and generate diverse warpage (elevation) samples for unseen designs.

| | |
|---|---|
| **Objective** | Capture how elevation distribution changes per design condition |
| **Objective** | Generate diverse samples covering the plausible distribution |
| **Objective** | Generalize to unseen design images |
| **Not Objective** | Minimize pixel-wise reconstruction error |

---

## 2. Data

### Input X — Elevation Image
- Grayscale image encoding continuous PCB surface height values (smooth gradient)
- Variable raw size (e.g., 413×284) → resized to **256×256**

### Condition C — Design Image
- Grayscale image with black design lines on white background (sparse, sharp edges)
- Variable raw size (e.g., 1481×1025) → resized to **256×256**
- Preprocessing note: apply dilation before resizing to preserve thin design lines

### Data Relationship
```
1 Design Image  :  N Elevation Images   (1:N mapping)
```
The model must learn the diversity of elevations per design as a distribution, not a single mapping.

### Dataset Scale

| | Current | Future |
|---|---|---|
| # Designs | 10 (diverse polygon shapes A–J) | 10+ (real PCB models) |
| Elevations per design | 300 (synthetic) | ~200 (real) |
| Total images | ~3,000 | ~2,000+ |

---

## 3. Validation Strategy

**Leave-One-Out Validation**
- Hold out 1 design entirely as the unseen test set
- Train on remaining N−1 designs
- Repeat for all N designs and average results

**Evaluation Metrics**

| Metric | Description |
|---|---|
| Diversity | Sample variance matches real elevation variance per design |
| Inter-condition Separability | Generated distributions differ across design conditions |
| Generalization | Meaningful distribution generated for unseen designs |
| FID / MMD | Distance between generated and real distributions |

---

## 4. Model Architecture

### Overview

```
Training

  Design Image (C)              Elevation Image (X)
        ↓                               ↓
  Design Encoder                Elevation Encoder
        ↓                               ↓
  condition vector c            (μ, σ) → z1
  [Deterministic, dim=64]       [Stochastic, dim=64]
        ↓                               ↓
        └──────── Fusion (A / B / C) ───┘
                        ↓
                     Decoder
                        ↓
              Reconstructed Elevation


Inference

  New Design Image → Design Encoder → c
  z1 ~ N(0, I)  ×  K samples
  Decoder → K generated elevation images
```

---

### Design Encoder

**Architecture: Lightweight Scratch CNN + Handcrafted Features**

```
Input (1×256×256)
→ Conv(32) → Conv(64) → Conv(128)
→ ChannelBottleneck(64, 1×1) → SpatialPool(4×4)   ← 16 spatial regions preserved
→ Flatten(1024) → MLP → c_cnn (dim=64)

Handcrafted Features (22-dim, computed at original resolution before resize):
  [0]    Global foreground density
  [1–4]  Quadrant densities (TL, TR, BL, BR)
  [5–6]  LR / TB asymmetry index
  [7–9]  H / V / D line orientation ratios
  [10]   Aspect ratio (H/W)
  [11]   Connected component count (log-normalised)
  [12]   Mean component area (normalised)
  [13–15] LR / TB / 180° symmetry scores (Pearson correlation)
  [16]   Centre density (inner 50 % area)
  [17]   Border density (outer ring)
  [18]   Radial mean distance from centre
  [19]   Perimeter ratio (edge pixels / foreground pixels)
  [20–21] Hu moments 1 & 2 (scale-invariant shape descriptors)
→ MLP(hidden=32) → c_hand (dim=32)

c = MLP(concat(c_cnn, c_hand))  →  dim=64  [Deterministic]
```

- Kept intentionally small to prevent memorizing few training designs
- Handcrafted features act as generalization anchors (computable for any unseen design)
- Feature selection: `python analyze_features.py --top 10 --write` identifies the most
  informative subset; the encoder uses only selected indices when `selected_features` is
  set in `config.txt`
- Strong regularization: Dropout + Weight Decay

---

### Elevation Encoder

**Architecture: Mid-size CNN + Reparameterization**

```
Input (1×256×256)
→ Conv(32) → Conv(64) → Conv(128) → Conv(256) → Flatten → MLP
→ μ (dim=64),  σ (dim=64)
→ z1 = μ + ε·σ,   ε ~ N(0, I)
```

- Larger capacity than design encoder; z1 quality directly determines sample diversity
- All stochasticity concentrated here; design encoder remains deterministic

---

### Fusion: z1 and c

Three strategies to be evaluated in order:

| | Method | Formula | When |
|---|---|---|---|
| **A** | Concatenation | `z = cat(z1, c)` | Always (baseline) |
| **B** | FiLM | `z = γ(c)·z1 + β(c)` | Primary recommendation |
| **C** | Cross-Attention | `z = Attn(Q=c, K=z1, V=z1)` | After data scale-up |

**Why not Self-Attention?**
Self-attention models relationships *within* a single representation (e.g., spatial positions within z1). Our need is to model the relationship *between* z1 and c — which is cross-attention's role. Self-attention is not applicable here.

---

### Decoder

```
z_fused
→ MLP → reshape (256×8×8)
→ [FiLM(c)] → Upsample + Conv(128)   8  → 16
→ [FiLM(c)] → Upsample + Conv(64)    16 → 32
→ [FiLM(c)] → Upsample + Conv(32)    32 → 64
→ [FiLM(c)] → Upsample + Conv(16)    64 → 128
→             Upsample + Conv(1) + Sigmoid   128 → 256
```

- FiLM applied at each upsample block so design condition persists throughout decoding
- Output activation: Sigmoid for [0,1] normalized elevation, Tanh for [−1,1]

---

## 5. Training Strategy

### Loss

```
Loss = Reconstruction Loss + β × KL Divergence

Reconstruction:  MSE(X_reconstructed, X)
KL:             −0.5 × Σ(1 + log σ² − μ² − σ²)
```

### β and KL Annealing

| | |
|---|---|
| β search range | 0.1 – 0.5 (reduced to prevent posterior collapse) |
| Annealing strategy | Cyclical: reset β → 0 periodically, then increase |
| Purpose | Prevent posterior collapse under small data |

### Augmentation

- **Design images:** rotation, flip, random crop, brightness jitter (aggressive)
- **Elevation images:** conservative — flip only if physically valid

---

## 6. Experiment Roadmap

| Step | Change | Goal |
|---|---|---|
| 1 | Concatenation-based CVAE | Establish baseline |
| 2 | Replace fusion with FiLM | Measure conditioning improvement |
| 3 | Add handcrafted features to design encoder | Measure generalization improvement |
| 4 | Tune β with cyclical annealing | Optimize distribution quality |
| 5 | Cross-attention fusion (post data scale-up) | Expand expressiveness |

### Current vs. Future Configuration

| | Current (10 synthetic designs) | Future (real data) |
|---|---|---|
| Design Encoder | Scratch CNN + 22 handcrafted features | Pretrained backbone (fine-tune) |
| Hand features | 22 (configurable subset via `selected_features`) | Re-rank with real data |
| Fusion | FiLM (baseline: Concat) | Cross-Attention |
| z2 from design | No (deterministic c) | Consider VAE-style z2 |
| Decoder | Transposed CNN + FiLM | Add U-Net skip connections |
| z1 dim | 64 | 128 |
| c dim | 64 | 64–128 |
| Sample diversity | Via temperature parameter | Tune with real distribution |

---

## 7. Synthetic Data Generation (for Development)

Due to data security restrictions, real PCB data cannot be used during development. Synthetic data must be generated to match the statistical and structural properties of real data.

---

### 7-1. Synthetic Design Image

**Goal:** Simulate a sparse grayscale PCB layout image (black lines on white background).

**Generation Rules:**
- Canvas: 256×256, white background (pixel value = 1.0)
- Draw random combinations of: horizontal lines, vertical lines, diagonal lines, rectangular pads, circular vias
- Line thickness: 1–3px (thin, as in real design images)
- Apply slight dilation after drawing to preserve line visibility after resize
- Generate **4 distinct design variants** (e.g., different line density, layout region, pad placement)

**Designs generated (10 variants, A–J):**

Each design is a closed polygon outline rendered in outline-only style (no fill),
auto-scaled to fill the canvas, and tiled in a symmetric grid.

| Design | Shape | Grid |
|---|---|---|
| A | Complex DIP — pin-slot notches on all sides | 2×2 |
| B | Winding staircase | 3×2 |
| C | Notched cross | 2×2 |
| D | Notched chamfered octagon | 2×3 |
| E–J | Additional shape variants (staircase, T, hourglass, comb, I-beam, U-channel) | 2×2 |

---

### 7-2. Synthetic Elevation Image

**Goal:** Simulate a smooth grayscale warpage map physically consistent with its paired design.

**Generation Rules:**
- Canvas: 256×256, pixel values in [0, 1] (continuous, smooth)
- Base shape: 2D Gaussian or sine surface to simulate physical bending
- Condition on design: regions with higher line density → slightly higher mean elevation (simulate thermal/mechanical stress)
- Add per-sample variation: random low-frequency noise (e.g., Perlin noise or sum of 2–3 random Gaussians) to simulate manufacturing variability
- Each design variant → generate **100–300 elevation samples** with controlled variance

**Variation sources to simulate:**
```
elevation = base_warp(design) + low_freq_noise() + small_random_tilt()
```

---

### 7-3. Dataset Summary (Synthetic)

| Design | # Elevation Samples | Warp Amplitude | Tilt Scale |
|---|---|---|---|
| A | 300 | 0.25 | 0.05 |
| B | 300 | 0.35 | 0.10 |
| C | 300 | 0.45 | 0.05 |
| D | 300 | 0.38 | 0.15 |
| E–J | 300 each | 0.28–0.42 | 0.06–0.15 |

- **Train:** 9 designs (leave-one-out rotation)
- **Test:** 1 unseen design
- Total: 3,000 image pairs

---

### 7-4. Generation Script Structure

See **Section 8 — Project Structure** for file locations.

---

## 8. Project Structure

```
project/
├── data_generation/
│   ├── generate_design.py       # Synthesize design images (10 variants, A–J)
│   ├── generate_elevation.py    # Synthesize elevation images (300 per design)
│   └── visualize_samples.py     # Sanity check: plot image pairs
├── data/
│   ├── design/                  # design_A.png … design_J.png
│   └── elevation/
│       ├── design_A/            # elevation images for design_A
│       ├── …
│       └── design_J/
├── models/
│   ├── design_encoder.py        # CNN + 22 handcrafted features → c
│   ├── elevation_encoder.py
│   ├── decoder.py
│   └── cvae.py
├── utils/
│   ├── dataset.py
│   ├── handcrafted_features.py  # 22-dim feature extractor
│   └── losses.py
├── train.py
├── evaluate.py
├── sample.py                    # supports --temperature for diversity control
└── analyze_features.py          # feature importance ranking and selection
```
