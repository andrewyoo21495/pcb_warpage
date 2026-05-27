# Latent Flow Matching (LFM) — PCB Warpage

## 1. 개요 및 동기

LFM은 LDM과 동일한 2-stage 전략을 사용하되, diffusion process 대신 **Flow Matching**을 사용합니다. Flow Matching은 source 분포(Gaussian)에서 target 분포(data)로의 **직선 경로(Optimal Transport path)**를 학습하는 방법입니다.

### Diffusion vs Flow Matching

```
Diffusion (LDM):
  z₀ ──(noise schedule)──► z_T     복잡한 경로, T steps
  z_T ──(DDIM reverse)───► z₀      α_bar, β 스케줄 필요

Flow Matching (LFM):
  z₀ ──(직선 경로)────────► z₁     단순한 직선, ODE
  z₀ ──(Euler ODE)────────► z₁     noise schedule 불필요
```

**장점**:
- Noise schedule (α, β, ᾱ) 전혀 불필요 → 구현이 단순
- 직선 경로이므로 더 적은 ODE step으로 충분
- Loss landscape가 smooth → 수렴이 빠름
- LDM과 동일한 CVAE latent space 활용

---

## 2. 수학적 정의

### Training

Flow Matching은 Gaussian N(0,I)에서 data 분포 p_data로의 ODE를 학습합니다:

```
dz/dt = v_θ(z_t, t, c),    t ∈ [0, 1]
```

**Linear interpolation (Optimal Transport path)**:
```
z₀ ~ N(0, I)                    # source (noise)
z₁ = μ = ElevationEncoder(x)    # target (data latent)
z_t = (1 - t) · z₀ + t · z₁     # interpolated point at time t
```

**Target velocity** (상수 — 직선 경로이므로):
```
v_target = z₁ - z₀ = dz_t/dt
```

**Training loss**:
```
L = E_{z₀, z₁, t} [ ‖v_θ(z_t, t, c) - v_target‖² ]
  = E_{z₀, z₁, t} [ ‖v_θ(z_t, t, c) - (z₁ - z₀)‖² ]
```

### Inference (Euler ODE solver)

```
z₀ ~ N(0, I) × temperature
dt = 1 / N_steps

for i = 0, 1, ..., N_steps - 1:
    t = i × dt
    v = v_θ(z_t, t, c)
    z_{t+dt} = z_t + v · dt

z₁ = z_{N_steps}    # generated latent
elevation = Decoder(FiLM(z₁, c), c)
```

---

## 3. 아키텍처 상세

### 3.1 전체 구조

LFM은 LDM과 동일한 CVAE 컴포넌트를 재사용합니다.
유일한 차이는 LatentDenoiser가 noise가 아닌 **velocity**를 예측한다는 점입니다.

```
┌──────────────────────────────────────────────────────┐
│  LatentFlowMatching                                  │
│                                                      │
│  ┌──────────────────────────┐  ❄️ Frozen            │
│  │  ElevationEncoder        │                       │
│  │  Design + Hand → c       │                       │
│  │  FiLM Fusion             │                       │
│  │  Decoder                 │                       │
│  └──────────────────────────┘                       │
│                                                      │
│  ┌──────────────────────────┐  🔥 Trainable         │
│  │  LatentDenoiser          │                       │
│  │  (velocity predictor)    │                       │
│  │  v_θ(z_t, t, c) → v     │                       │
│  └──────────────────────────┘                       │
└──────────────────────────────────────────────────────┘
```

### 3.2 Velocity Network

LatentDenoiser와 동일한 구조를 사용하되, 입력 timestep이 연속값 t ∈ [0,1]:

```
t (float, [0,1]) → SinusoidalEmbedding(128) → MLP → t_emb (512)
[z_t, c] → Linear(64+c_dim → 512) → h

AdaLN-ResBlock × 8:
  h = LN(h) * (1 + scale(t_emb)) + shift(t_emb)
  h = SiLU → Linear → Dropout → ... + skip

LayerNorm → Linear(512→64) → v_pred
```

### 3.3 LDM vs LFM 학습 비교

```
                    LDM                          LFM
                ─────────────────       ─────────────────
Timestep        t ∈ {0, ..., T-1}       t ∈ [0, 1] (continuous)
                (integer)                (float)

Input z_t       √ᾱ_t · z₀              (1-t) · z₀ + t · z₁
                + √(1−ᾱ_t) · ε

Target          ε (noise)               z₁ - z₀ (velocity)

Loss            MSE(ε_pred, ε)          MSE(v_pred, z₁-z₀)

Schedule        cosine β schedule       없음 ✓
                (α, β, ᾱ 필요)

Inference       DDIM (50 steps)         Euler ODE (30 steps)
```

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
│  z₀ ~ N(0,I)    │            │
│  × temperature   │            │
│  (64-dim)        │            │
└────────┬────────┘            │
         │                      │
         ▼                      │
┌─────────────────┐            │
│  Euler ODE       │            │
│  (30 steps)      │◄──── c ───┤
│  t: 0 → 1       │            │
│                  │            │
│  z ← z + v·dt   │            │
│  v = v_θ(z,t,c) │            │
└────────┬────────┘            │
         │ z₁ (64-dim)         │
         │                      │
         ▼                      │
┌─────────────────┐            │
│  FiLM Fusion     │ ❄️         │
│  γ(c)·z₁ + β(c) │◄──── c ───┘
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
| `model_type` | `lfm` | 모델 타입 |
| `cvae_checkpoint` | `./outputs/cvae_pcb.pth` | 사전학습된 CVAE 체크포인트 경로 |
| `lfm_ode_steps` | 30 | Euler ODE inference steps |
| `lfm_hidden_dim` | 512 | Velocity net MLP hidden width |
| `lfm_n_blocks` | 8 | Velocity net residual block 수 |
| `lfm_dropout` | 0.1 | Velocity net dropout rate |
| `lfm_sigma_min` | 0.001 | 수치 안정성 (t의 boundary clamp) |
| `lfm_finetune_encoder` | False | Design encoder fine-tuning 여부 |
| `ema_decay` | 0.9999 | EMA decay rate |

---

## 6. 사용법

### Stage 1: CVAE 사전학습

```bash
python train.py --config config.txt    # model_type = cvae
```

### Stage 2: LFM 학습

```bash
python train.py --config config_lfm.txt
```

### Sampling

```bash
python sample.py --config config_lfm.txt --design data/design/design_A.png --k 16
python sample.py --config config_lfm.txt --design-dir data/design/ --k 16
```

### Evaluation

```bash
python evaluate.py --config config_lfm.txt
```

---

## 7. LDM vs LFM 선택 가이드

| 기준 | LDM | LFM |
|------|-----|-----|
| **구현 복잡도** | 보통 (noise schedule 필요) | 낮음 (noise schedule 불필요) |
| **수렴 안정성** | 높음 | 매우 높음 |
| **Inference 속도** | 50 steps (DDIM) | 30 steps (Euler) |
| **생성 품질** | 우수 | 우수 (더 적은 step으로 유사 품질) |
| **이론적 기반** | DDPM (well-established) | Flow Matching (newer, simpler) |
| **하이퍼파라미터** | T, eta, ddim_steps | ode_steps, sigma_min |

**권장 전략**: LDM 먼저 학습 → 결과 확인 → LFM으로 전환하여 비교

---

## 8. Flow Matching 이론적 배경

### Continuous Normalizing Flow (CNF)

Flow Matching은 Continuous Normalizing Flow의 학습 방법입니다:

```
dz/dt = v_θ(z, t)
```

이 ODE는 시간 t=0의 분포 p₀(z)를 t=1의 분포 p₁(z)로 변환합니다.

### Conditional Flow Matching (CFM)

CFM은 각 데이터 포인트 z₁에 대해 conditional path를 정의합니다:

```
p_t(z | z₁) = N(z; t·z₁, (1-(1-σ_min)·t)²·I)
```

이 conditional path의 velocity field를 평균하면 marginal velocity field를 얻습니다:

```
v_t(z) = E_{z₁~p_data}[ u_t(z|z₁) · p_t(z|z₁) / p_t(z) ]
```

**핵심 장점**: conditional velocity u_t(z|z₁) = z₁ - z₀는 closed-form이므로,
regression으로 직접 학습할 수 있습니다.

### 참고 논문

- Lipman et al. (2023) "Flow Matching for Generative Modeling"
- Tong et al. (2023) "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport"

---

## 9. 파일 구조

```
models/
├── latent_denoiser.py    # LatentDenoiser (MLP with AdaLN) — LDM과 공유
├── lfm.py                # LatentFlowMatching (main class)
├── design_encoder.py     # DesignEncoder (CVAE에서 재사용, frozen)
├── elevation_encoder.py  # ElevationEncoder (CVAE에서 재사용, frozen)
└── decoder.py            # Decoder (CVAE에서 재사용, frozen)
```
