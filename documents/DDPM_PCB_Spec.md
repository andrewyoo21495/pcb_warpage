# PCB Elevation Image Generation System — Project Specification

---

## 1. Problem Definition

Given a **design image** of a PCB (Printed Circuit Board), build a conditional generative model that produces multiple **elevation images** representing the board's warpage distribution for that design.

The primary goal is **not** pixel-perfect reconstruction, but rather learning how the elevation distribution shifts with design conditions, and estimating that distribution for new (unseen) designs.

---

## 2. Data Description

### 2.1 Elevation Image (Generation Target)

| Item | Detail |
|------|--------|
| Meaning | Grayscale height map representing PCB surface warpage |
| Raw format | Numeric measurements in txt format |
| Preprocessing | Downsampling → null value interpolation → grayscale image conversion |
| Size | Varies (e.g., 413×284, 499×273), **resized to 256×256** for training |
| Channels | 1ch (grayscale) |
| Value range | 0–255 (or [-1, 1] after normalization) |

### 2.2 Design Image (Condition Input)

| Item | Detail |
|------|--------|
| Meaning | PCB layout diagram — circuit patterns drawn in black on a white background |
| Size | Varies (e.g., 1481×1025), **resized to 256×256** (same as elevation) for training |
| Channels | 1ch (grayscale) |
| Characteristics | Currently similar patterns with partial variations (same product model). More diverse models planned in the future |

### 2.3 Data Scale and Structure

```
data/
├── design_A/          # Design type 1
│   ├── design.png
│   ├── metadata.json  # Original dimensions, design density, etc.
│   └── elevations/    # ~100–300 images
│       ├── elev_001.png
│       ├── elev_002.png
│       └── ...
├── design_B/          # Design type 2
├── design_C/          # Design type 3
└── design_D/          # Design type 4
```

| Item | Current | Future Target |
|------|---------|---------------|
| Number of design types | 4 | 10+ |
| Elevations per design | 100–300 | ~200/type |
| Total data size | ~400–1,200 | ~2,000+ |

### 2.4 Key Data Characteristics

- **1:N relationship**: One design maps to many elevations. Identical designs produce varying warpage patterns in real manufacturing, so the model must learn a **conditional distribution**, not a deterministic mapping.
- **Small dataset**: Few design types with only hundreds of samples each — overfitting prevention is critical.
- **Inter-design similarity**: Current designs are mostly minor variations of the same product model, but more heterogeneous designs will be added over time.

---

## 3. Condition Features

Condition information injected into the model falls into two categories.

### 3.1 Spatial Condition

- The resized design image itself (256×256, 1ch)
- Extracted as multi-scale feature maps via a CNN encoder → injected at matching U-Net resolution levels

### 3.2 Global Condition

Scalar/vector features derived from the design image:

| Feature | Description | Note |
|---------|-------------|------|
| `original_design_h` | Original design image height (px) | Normalize |
| `original_design_w` | Original design image width (px) | Normalize |
| `aspect_ratio` | h/w ratio | |
| `original_elev_h` | Original elevation image height (px) | Normalize |
| `original_elev_w` | Original elevation image width (px) | Normalize |
| `design_density` | Black pixel ratio in the design image | Proxy for circuit density |
| *(extensible)* | Edge density, line width distribution, regional density, etc. | Domain-knowledge-driven additions |

→ Embedded via MLP into a global condition vector

---

## 4. Model Architecture

### 4.1 Selected Model: Conditional DDPM (Denoising Diffusion Probabilistic Model)

**Rationale:**

| Alternative | Why Not |
|-------------|---------|
| CVAE | Blurry outputs, poor spatial detail preservation |
| cGAN | Mode collapse risk — fatal for distribution learning objective |
| Flow | Implementation complexity not justified for small data regime |
| **DDPM** | **Superior mode coverage, stable with small data, optimal for full distribution learning** |

### 4.2 Architecture Overview

```
Design Image (256×256, 1ch)
    │
    ▼
Condition Encoder (lightweight CNN)
    ├── Spatial features: multi-scale feature maps (128², 64², 32²)
    └── Global features: pooled vector + extra scalar features (via MLP)

Noise z_t (256×256, 1ch)
    │
    ▼
U-Net Noise Predictor
    ├── Spatial condition: channel-wise concat at matching resolutions
    ├── Global condition + Timestep: AdaGN (Adaptive Group Normalization) injection
    │
    ▼
Predicted noise ε_θ
    │
    (Reverse diffusion × T steps)
    │
    ▼
Generated Elevation Image (256×256, 1ch)
```

### 4.3 Key Design Choices

| Item | Choice | Rationale |
|------|--------|-----------|
| β schedule | Cosine | More stable with small datasets |
| Timesteps (T) | 1000 (train), DDIM 50 steps (inference) | Training stability + inference speed |
| Condition injection | Spatial concat + AdaGN | Captures both spatial structure and global attributes |
| Model size | base_ch=64, ~10M params | Prevents overfitting given data scale |
| Sampling | DDIM, eta=0.5–0.8 | Balances diversity and quality |

---

## 5. Training Strategy

### 5.1 Data Augmentation

- **Geometric**: HorizontalFlip, VerticalFlip (applied **identically** to both elevation and design to preserve spatial correspondence)
- **Elevation-only**: Light Gaussian noise injection (optional)
- **Caution**: No color jitter (grayscale data). Never apply augmentations that break spatial alignment between elevation and design.

### 5.2 Regularization

| Technique | Setting |
|-----------|---------|
| EMA (Exponential Moving Average) | decay=0.9999, essential for generation quality |
| Dropout | 0.1–0.2 |
| Weight Decay | 1e-4 |
| Model size constraint | base_ch=64 |

### 5.3 Validation Strategy

- **Leave-one-design-out**: Hold out 1 of 4 design types entirely → evaluate generalization to unseen designs
- 4-fold cross-validation across all design types for comprehensive assessment
- This mirrors the actual deployment scenario

### 5.4 Hyperparameters (Baseline)

```yaml
batch_size: 16
learning_rate: 1e-4
weight_decay: 1e-4
ema_decay: 0.9999
epochs: 2000        # Small data → requires sufficient epochs
T: 1000
dropout: 0.15
optimizer: AdamW
image_size: 256
```

---

## 6. Inference Pipeline

```
Input:
  - New (unseen) design image
  - Metadata (original dimensions, etc.)
  - Number of samples to generate: n

Process:
  1. Design image → resize to 256×256, normalize
  2. Condition Encoder → spatial features + global features (computed once)
  3. Sample n Gaussian noise tensors
  4. DDIM reverse process (50 steps, eta=0.5–0.8)
  5. Denormalize generated images → optionally resize to target dimensions

Output:
  - n elevation images (256×256 or original size)
  - Pixel-wise statistics (mean, std maps, etc.) can be derived from these
```

---

## 7. Evaluation Metrics

| Metric | Purpose | Note |
|--------|---------|------|
| Per-pixel mean/std map comparison | Intuitive distribution characterization | Generated n samples vs. real data |
| FID (Fréchet Inception Distance) | Quantitative distribution distance | Limited reliability with small data |
| Coverage & Density | Measures how well generated samples span the real distribution | |
| Visual inspection | Spatial coherence between design patterns and elevation | Domain expert assessment |

---

## 8. Future Roadmap (Upon Data Expansion)

When 10+ design types and 2,000+ total samples are available:

| Improvement | Detail |
|-------------|--------|
| Stronger condition encoder | Lightweight CNN → pretrained backbone (e.g., ResNet-18) |
| Cross-attention injection | Replace spatial concat with cross-attention for more flexible conditioning |
| Classifier-free guidance | Drop condition 10–20% during training, control guidance scale at inference |
| Latent diffusion | Move from pixel-space to VAE latent-space diffusion for efficiency |
| Higher resolution | 256×256 → 512×512+ |

---

## 9. Constraints & Assumptions

- Elevation and design images are assumed to be **spatially aligned**.
- Information loss from resizing to 256×256 is acceptable.
- **Distribution diversity and conditional variation** take priority over reconstruction fidelity.
- At the current data scale, model capacity is intentionally limited to prioritize overfitting prevention.
