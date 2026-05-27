# Latent Diffusion Model (LDM) — PCB Warpage

## 1. 개요 및 동기

기존 pixel-space DDPM은 128×128 이미지 공간에서 직접 diffusion을 수행하여, ~2,000개의 소규모 학습 데이터로는 15,000 epoch 이후에도 수렴하지 못하는 문제가 있었습니다.

**핵심 관찰**: CVAE 모델은 동일한 데이터에서 64차원 latent space를 성공적으로 학습하여 양질의 reconstruction을 달성했습니다.

**해결 전략**: CVAE가 학습한 의미있는 64차원 latent space를 활용하여, pixel space(16,384차원) 대신 latent space(64차원)에서 diffusion을 수행합니다. 이로써 학습 난이도가 극적으로 감소합니다.

---

## 2. 2-Stage 학습 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: CVAE 사전학습 (기존 CVAE 학습과 동일)             │
│                                                              │
│  Elevation (256×256) → ElevationEncoder → μ, logvar → z₀   │
│  Design (256×256)    → DesignEncoder    → c                 │
│  z_fused = FiLM(z₀, c) → Decoder → Reconstruction          │
│                                                              │
│  * 이 단계에서 encoder/decoder가 의미있는 latent space 학습  │
│  * 학습 완료 후 checkpoint 저장                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Latent Diffusion 학습 (새로 추가된 부분)          │
│                                                              │
│  ┌──────────────────────────────┐                           │
│  │  Frozen CVAE Components     │                           │
│  │  ├─ ElevationEncoder ❄️      │                           │
│  │  ├─ DesignEncoder ❄️         │                           │
│  │  ├─ FiLM Fusion ❄️          │                           │
│  │  └─ Decoder ❄️              │                           │
│  └──────────────────────────────┘                           │
│  ┌──────────────────────────────┐                           │
│  │  Trainable: LatentDenoiser  │  ← MLP (~2M params)      │
│  │  ε_θ(z_t, t, c) → noise    │                           │
│  └──────────────────────────────┘                           │
│                                                              │
│  학습: z₀ = μ (frozen encoder 출력)                         │
│        z_t = √ᾱ_t · z₀ + √(1−ᾱ_t) · ε                    │
│        Loss = MSE(ε_θ(z_t, t, c), ε)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 아키텍처 상세

### 3.1 LatentDenoiser (MLP 기반 noise predictor)

64차원 latent space에서 동작하므로 UNet 대신 경량 MLP를 사용합니다.

```
┌──────────────────────────────────────────────────────┐
│                  LatentDenoiser                       │
│                                                      │
│  Inputs:                                             │
│    z_t : (B, 64)     noisy latent                   │
│    t   : (B,)        timestep (integer, 0~T-1)      │
│    c   : (B, c_dim)  condition vector               │
│                                                      │
│  ┌────────────────────────────────────────────┐     │
│  │  t → SinusoidalEmbedding(128)              │     │
│  │    → Linear(128→512) → SiLU               │     │
│  │    → Linear(512→512)         → t_emb       │     │
│  └────────────────────────────────────────────┘     │
│                                                      │
│  ┌────────────────────────────────────────────┐     │
│  │  [z_t, c] → Linear(64+c_dim → 512) → h    │     │
│  └────────────────────────────────────────────┘     │
│                                                      │
│  ┌────────────────────────────────────────────┐     │
│  │  AdaLN-ResBlock × 8 (conditioned on t_emb) │     │
│  │                                             │     │
│  │  Each block:                                │     │
│  │    AdaLN(h, t_emb):                        │     │
│  │      LN(h) * (1 + scale(t_emb)) + shift   │     │
│  │    → SiLU → Linear(512→512) → Dropout      │     │
│  │    → AdaLN(h, t_emb)                       │     │
│  │    → SiLU → Linear(512→512) → Dropout      │     │
│  │    + residual skip connection               │     │
│  └────────────────────────────────────────────┘     │
│                                                      │
│  ┌────────────────────────────────────────────┐     │
│  │  LayerNorm → Linear(512→64) → ε_pred       │     │
│  └────────────────────────────────────────────┘     │
│                                                      │
│  Output: ε_pred (B, 64)                             │
└──────────────────────────────────────────────────────┘
```

### 3.2 Noise Schedule

DDPM과 동일한 cosine beta schedule을 사용하되, T를 500으로 줄임:

```
β_t = 1 - (ᾱ_t / ᾱ_{t-1})

ᾱ_t = ∏_{s=1}^{t} α_s = ∏_{s=1}^{t} (1 - β_s)

f(t) = cos²((t/T + s) / (1+s) · π/2),  s = 0.008
```

T=500이면 pixel DDPM(T=1000) 대비 절반의 timestep으로 충분합니다.
Latent space가 64차원으로 매우 작기 때문입니다.

### 3.3 DDIM 샘플링

Inference 시 DDIM (Denoising Diffusion Implicit Models) 사용:

```
z_T ~ N(0, I) × temperature

for t in reversed(ddim_schedule):
    ε = ε_θ(z_t, t, c)
    z₀_pred = (z_t - √(1−ᾱ_t) · ε) / √ᾱ_t
    σ = η · √((1−ᾱ_{t-1})/(1−ᾱ_t)) · √(1−ᾱ_t/ᾱ_{t-1})
    z_{t-1} = √ᾱ_{t-1} · z₀_pred + dir + σ · noise

z_fused = FiLM(z₀, c)
elevation = Decoder(z_fused, c)    # → (K, 1, 256, 256)
```

LDM에서는 η=0 (deterministic DDIM)을 기본값으로 사용합니다.
Latent space에서는 deterministic sampling이 더 안정적인 결과를 제공합니다.

---

## 4. 전체 Inference 흐름

```
New Design Image (256×256)
        │
        ▼
┌─────────────────┐
│  DesignEncoder   │ ❄️ frozen
│  (CNN + Hand)    │
└────────┬────────┘
         │ c (c_dim)
         │
         ├──────────────────────┐
         │                      │
         ▼                      │
┌─────────────────┐            │
│  z_T ~ N(0,I)   │            │
│  × temperature   │            │
│  (64-dim)        │            │
└────────┬────────┘            │
         │                      │
         ▼                      │
┌─────────────────┐            │
│  DDIM Reverse    │            │
│  (50 steps)      │◄──── c ───┤
│  ε_θ(z_t, t, c) │            │
└────────┬────────┘            │
         │ z₀ (64-dim)         │
         │                      │
         ▼                      │
┌─────────────────┐            │
│  FiLM Fusion     │ ❄️         │
│  γ(c)·z₀ + β(c) │◄──── c ───┘
└────────┬────────┘
         │ z_fused (64-dim)
         ▼
┌─────────────────┐
│  Decoder ❄️      │
│  FC → Upsample   │
│  × 5 blocks      │
│  + Sigmoid        │
└────────┬────────┘
         │
         ▼
Elevation Image (256×256)
```

---

## 5. Config 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `model_type` | `ldm` | 모델 타입 |
| `cvae_checkpoint` | `./outputs/cvae_pcb.pth` | 사전학습된 CVAE 체크포인트 경로 |
| `ldm_T` | 500 | Latent space diffusion timesteps |
| `ldm_ddim_steps` | 50 | DDIM inference steps |
| `ldm_hidden_dim` | 512 | Denoiser MLP hidden width |
| `ldm_n_blocks` | 8 | Denoiser residual block 수 |
| `ldm_dropout` | 0.1 | Denoiser dropout rate |
| `ldm_finetune_encoder` | False | Design encoder fine-tuning 여부 |
| `ema_decay` | 0.9999 | EMA decay rate |
| `z_dim` | 64 | Latent dimension (CVAE와 동일해야 함) |
| `c_dim` | 64 | Condition dimension (CVAE와 동일해야 함) |
| `fusion_method` | film | Fusion method (CVAE와 동일해야 함) |

---

## 6. 사용법

### Stage 1: CVAE 사전학습

```bash
python train.py --config config.txt    # model_type = cvae
```

### Stage 2: LDM 학습

```bash
python train.py --config config_ldm.txt
```

### Sampling

```bash
python sample.py --config config_ldm.txt --design data/design/design_A.png --k 16
python sample.py --config config_ldm.txt --design-dir data/design/ --k 16
```

### Evaluation

```bash
python evaluate.py --config config_ldm.txt
```

---

## 7. DDPM 대비 장점

| 비교 항목 | Pixel DDPM | LDM |
|-----------|-----------|-----|
| **Diffusion 공간** | 128×128 = 16,384차원 | 64차원 |
| **학습 난이도** | 매우 높음 | 낮음 |
| **필요 데이터 수** | 많음 (>10K 권장) | 적음 (~2K 가능) |
| **모델 크기** | ~14.3M (UNet) | ~2M (MLP denoiser) |
| **수렴 속도** | 느림 (15K epoch에도 미수렴) | 빠름 (500 epoch 이내 기대) |
| **Inference 속도** | 50 DDIM steps × UNet | 50 DDIM steps × MLP (훨씬 빠름) |
| **추가 요구사항** | 없음 | 사전학습된 CVAE 필요 |

---

## 8. 파일 구조

```
models/
├── latent_denoiser.py    # LatentDenoiser (MLP with AdaLN)
├── ldm.py                # LatentDiffusionModel (main class)
├── design_encoder.py     # DesignEncoder (CVAE에서 재사용, frozen)
├── elevation_encoder.py  # ElevationEncoder (CVAE에서 재사용, frozen)
└── decoder.py            # Decoder (CVAE에서 재사용, frozen)
```
