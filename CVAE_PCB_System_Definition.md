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
| # Designs | 4 (similar layout variants) | 10+ (diverse PCB models) |
| Elevations per design | 100–300 | ~200 |
| Total images | ~400–1,200 | ~2,000+ |

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
→ Conv(32) → Conv(64) → Conv(128) → GlobalAvgPool → MLP → c_cnn (dim=64)

Handcrafted Features (computed separately):
  - Line density per region (black pixel ratio)
  - Left-right / top-bottom asymmetry index
  - Line orientation distribution (horizontal / vertical / diagonal)
→ MLP → c_hand (dim=16)

c = MLP(concat(c_cnn, c_hand))  →  dim=64  [Deterministic]
```

- Kept intentionally small to prevent memorizing 4 training designs
- Handcrafted features act as generalization anchors (computable for any unseen design)
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
| β search range | 1.0 – 4.0 |
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

| | Current (4 designs) | Future (10+ designs) |
|---|---|---|
| Design Encoder | Scratch CNN + Handcrafted | Pretrained backbone (fine-tune) |
| Fusion | FiLM (baseline: Concat) | Cross-Attention |
| z2 from design | No (deterministic c) | Consider VAE-style z2 |
| Decoder | Transposed CNN + FiLM | Add U-Net skip connections |
| z1 dim | 64 | 128 |
| c dim | 64 | 64–128 |

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

**Handcrafted feature targets per design variant:**

| Design | Line Density | Asymmetry | Dominant Orientation |
|---|---|---|---|
| A | Low (~5%) | Low | Horizontal |
| B | Medium (~15%) | Medium (left-heavy) | Mixed |
| C | High (~30%) | Low | Vertical |
| D | Medium (~20%) | High (top-heavy) | Mixed |

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

| Design | # Elevation Samples | Notes |
|---|---|---|
| A | 200 | Low density, mild warp |
| B | 200 | Medium density, asymmetric warp |
| C | 200 | High density, strong warp |
| D | 200 | Medium density, tilted warp |

- **Train:** 3 designs (leave-one-out rotation)
- **Test:** 1 unseen design
- Total: 800 image pairs

---

### 7-4. Generation Script Structure

See **Section 8 — Project Structure** for file locations.

---

## 8. Project Structure

```
project/
├── data_generation/
│   ├── generate_design.py       # Synthesize design images (4 variants)
│   ├── generate_elevation.py    # Synthesize elevation images per design
│   └── visualize_samples.py     # Sanity check: plot image pairs
├── data/
│   ├── design/                  # design_A.png, design_B.png, ...
│   └── elevation/
│       ├── design_A/            # elevation images for design_A
│       ├── design_B/
│       └── ...
├── models/
│   ├── design_encoder.py
│   ├── elevation_encoder.py
│   ├── decoder.py
│   └── cvae.py
├── utils/
│   ├── dataset.py
│   ├── handcrafted_features.py
│   └── losses.py
├── train.py
├── evaluate.py
└── sample.py
```
