# DF²M Usage Guide

DF²M (Decomposed Fourier-Conditioned Flow Matching) 파이프라인의 실행 가이드.

두 가지 실행 방식을 지원합니다:
- **Shell Script 방식** — 다수의 fold를 여러 GPU에 자동 분배하여 병렬 실행
- **Python 직접 실행 방식** — 단일 fold/phase를 세밀하게 제어하며 실행

---

## 목차

1. [사전 준비](#1-사전-준비)
2. [프로젝트 구조](#2-프로젝트-구조)
3. [Config 설정](#3-config-설정)
4. [방법 A: Shell Script로 전체 파이프라인 실행](#4-방법-a-shell-script로-전체-파이프라인-실행)
5. [방법 B: Python으로 직접 실행](#5-방법-b-python으로-직접-실행)
6. [GPU 설정 옵션 상세](#6-gpu-설정-옵션-상세)
7. [평가 및 샘플링](#7-평가-및-샘플링)
8. [검증 스크립트](#8-검증-스크립트)
9. [체크포인트 구조](#9-체크포인트-구조)
10. [트러블슈팅](#10-트러블슈팅)

---

## 1. 사전 준비

```bash
# 필수 패키지
pip install torch torchvision numpy scipy Pillow matplotlib

# Python 버전: 3.10+
python --version

# GPU 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

프로젝트 루트 디렉토리 (`pcb_warpage/`)에서 모든 명령을 실행합니다.

---

## 2. 프로젝트 구조

```
dfm_approach/
├── config_dfm.txt              # 하이퍼파라미터 설정 파일
├── USAGE.md                    # 이 문서
├── architecture_overview.html  # 아키텍처 문서 (SVG 다이어그램 포함)
│
├── models/                     # 모델 구현
│   ├── __init__.py             # build_dfm_models() 팩토리
│   ├── fno_mean_predictor.py   # Module A: FNO
│   ├── condition_encoder.py    # 디자인 조건 인코더
│   ├── residual_cae.py         # Module B-1: 잔차 CAE
│   ├── velocity_net.py         # AdaLN-ResBlock 속도장 네트워크
│   └── ot_cfm.py               # Module B-2: OT-CFM
│
├── utils/
│   ├── dfm_dataset.py          # 데이터셋 및 데이터로더
│   └── dfm_losses.py           # 손실 함수
│
├── train.py                    # 학습 스크립트
├── evaluate.py                 # 평가 스크립트
├── sample.py                   # 샘플링/추론 스크립트
├── verify.py                   # 서버 배포 검증 스크립트
│
├── run_pipeline.sh             # 전체 파이프라인 (Shell)
├── run_all_folds.sh            # Fold 분배 실행 (Shell)
└── run_evaluate_all.sh         # 전체 Fold 평가 (Shell)
```

---

## 3. Config 설정

`config_dfm.txt`를 환경에 맞게 수정합니다:

```bash
# 반드시 확인/수정해야 하는 항목:
design_image_dir        /data2/AOI/B8_AOI_preprocessed   # 디자인 이미지 경로
elevation_base_dir      /data2/AOI/B8_AOI_preprocessed   # elevation 이미지 경로
design_names            B8+REV02+..., B8+REV03+...       # 디자인 이름 목록
gpu_ids                 0                                 # 기본 GPU 인덱스
```

---

## 4. 방법 A: Shell Script로 전체 파이프라인 실행

### 4-1. 전체 파이프라인 (권장)

모든 fold에 대해 Phase 1 → 2 → 3 → 평가를 자동 실행합니다.

```bash
# 기본: GPU 8개 사용
bash dfm_approach/run_pipeline.sh --gpus 8

# GPU 4개만 사용
bash dfm_approach/run_pipeline.sh --gpus 4

# GPU 2개 사용
bash dfm_approach/run_pipeline.sh --gpus 2
```

**동작 방식:**
1. Phase 1 (FNO) 모든 fold → GPU에 라운드-로빈 분배 → 병렬 실행
2. Phase 1 완료 후 Phase 2 (CAE) 동일하게 실행
3. Phase 2 완료 후 Phase 3 (OT-CFM) 동일하게 실행
4. 모든 학습 완료 후 평가 실행

**GPU 분배 예시** (10 folds, 4 GPUs):
```
GPU 0: fold 0, fold 4, fold 8   (순차 실행)
GPU 1: fold 1, fold 5, fold 9   (순차 실행)
GPU 2: fold 2, fold 6           (순차 실행)
GPU 3: fold 3, fold 7           (순차 실행)
← 4개 GPU 간에는 병렬 실행 →
```

### 4-2. 특정 Phase부터 재개

```bash
# Phase 1(FNO) 이미 완료 → Phase 2부터 시작
bash dfm_approach/run_pipeline.sh --gpus 4 --skip-to 2

# Phase 1, 2 모두 완료 → Phase 3만 실행
bash dfm_approach/run_pipeline.sh --gpus 4 --skip-to 3

# 특정 Phase만 단독 실행
bash dfm_approach/run_pipeline.sh --gpus 4 --phase-only 1
bash dfm_approach/run_pipeline.sh --gpus 4 --phase-only 2
bash dfm_approach/run_pipeline.sh --gpus 4 --phase-only 3
```

### 4-3. 평가 건너뛰기

```bash
# 학습만 하고 평가는 나중에
bash dfm_approach/run_pipeline.sh --gpus 8 --skip-eval
```

### 4-4. 개별 fold 분배 스크립트

```bash
# Phase 1만, GPU 0~3에서 실행
bash dfm_approach/run_all_folds.sh --phase 1 --gpus 4

# Phase 2만, GPU 4~7에서 실행 (오프셋 사용)
bash dfm_approach/run_all_folds.sh --phase 2 --gpus 4 --gpu-offset 4

# 평가만, GPU 8개
bash dfm_approach/run_evaluate_all.sh --gpus 8 --k 50
```

### 4-5. Shell Script 전체 옵션 요약

| Script | 옵션 | 설명 |
|--------|------|------|
| `run_pipeline.sh` | `--gpus N` | 사용할 GPU 수 (기본: 8) |
| | `--skip-to P` | Phase P부터 시작 (1, 2, 3) |
| | `--phase-only P` | Phase P만 실행 |
| | `--skip-eval` | 평가 단계 건너뛰기 |
| | `--config PATH` | config 파일 경로 |
| `run_all_folds.sh` | `--phase P` | 실행할 Phase (0=전부, 1, 2, 3) |
| | `--gpus N` | 사용할 GPU 수 (기본: 8) |
| | `--gpu-offset G` | 시작 GPU 인덱스 (기본: 0) |
| | `--config PATH` | config 파일 경로 |
| `run_evaluate_all.sh` | `--gpus N` | 사용할 GPU 수 |
| | `--k K` | fold당 생성할 샘플 수 |
| | `--config PATH` | config 파일 경로 |

---

## 5. 방법 B: Python으로 직접 실행

Shell script 없이 Python 스크립트를 직접 호출합니다.

### 5-1. 전체 파이프라인 (단일 fold)

```bash
# fold 0 학습: Phase 1 → 2 → 3 순차 실행
python dfm_approach/train.py \
    --config dfm_approach/config_dfm.txt \
    --val_fold 0
```

### 5-2. Phase별 개별 실행

```bash
# Phase 1: FNO Mean Predictor 학습
python dfm_approach/train.py \
    --config dfm_approach/config_dfm.txt \
    --phase 1 \
    --val_fold 0 \
    --gpu 0

# Phase 2: Residual CAE 학습 (Phase 1 체크포인트 필요)
python dfm_approach/train.py \
    --config dfm_approach/config_dfm.txt \
    --phase 2 \
    --val_fold 0 \
    --gpu 0

# Phase 3: OT-CFM 학습 (Phase 1, 2 체크포인트 필요)
python dfm_approach/train.py \
    --config dfm_approach/config_dfm.txt \
    --phase 3 \
    --val_fold 0 \
    --gpu 0
```

### 5-3. Multi-GPU DataParallel (단일 프로세스)

단일 fold의 학습을 여러 GPU에 분산하여 배치 병렬처리합니다.

```bash
# GPU 0,1,2,3 총 4개로 DataParallel 학습
python dfm_approach/train.py \
    --config dfm_approach/config_dfm.txt \
    --val_fold 0 \
    --gpu 0 \
    --num-gpus 4

# GPU 4번부터 시작하여 GPU 4,5,6,7 사용
python dfm_approach/train.py \
    --config dfm_approach/config_dfm.txt \
    --val_fold 0 \
    --gpu 4 \
    --num-gpus 4

# Phase 1만 2개 GPU로
python dfm_approach/train.py \
    --config dfm_approach/config_dfm.txt \
    --phase 1 \
    --val_fold 0 \
    --gpu 0 \
    --num-gpus 2
```

**`--num-gpus` 동작 방식:**
- `--gpu G --num-gpus N` → GPU G, G+1, ..., G+N-1 사용
- PyTorch `DataParallel`로 배치를 GPU에 분산
- 체크포인트는 항상 단일 모델 state_dict으로 저장 (DP wrapper 자동 해제)
- 평가/샘플링은 단일 GPU로 충분하므로 `--num-gpus`는 학습 시에만 효과

### 5-4. Tag를 이용한 fold별 체크포인트 분리

```bash
# fold별로 별도 체크포인트 파일 생성
python dfm_approach/train.py --val_fold 0 --tag fold0 --gpu 0
python dfm_approach/train.py --val_fold 1 --tag fold1 --gpu 1
python dfm_approach/train.py --val_fold 2 --tag fold2 --gpu 2
```

`--tag fold0` 사용 시 체크포인트 파일명:
```
outputs/dfm_fno_fold0.pth    (기본: outputs/dfm_fno.pth)
outputs/dfm_cae_fold0.pth    (기본: outputs/dfm_cae.pth)
outputs/dfm_cfm_fold0.pth    (기본: outputs/dfm_cfm.pth)
outputs/train_dfm_fold0.log  (기본: outputs/train_dfm.log)
```

### 5-5. 수동 병렬 실행 (shell script 없이)

여러 터미널 또는 백그라운드에서 fold를 병렬로 수동 실행:

```bash
# 터미널 1: fold 0, 1을 GPU 0에서 순차 실행
python dfm_approach/train.py --val_fold 0 --tag fold0 --gpu 0 && \
python dfm_approach/train.py --val_fold 1 --tag fold1 --gpu 0

# 터미널 2: fold 2, 3을 GPU 1에서 순차 실행
python dfm_approach/train.py --val_fold 2 --tag fold2 --gpu 1 && \
python dfm_approach/train.py --val_fold 3 --tag fold3 --gpu 1

# 또는 백그라운드로 실행
python dfm_approach/train.py --val_fold 0 --tag fold0 --gpu 0 > logs/fold0.log 2>&1 &
python dfm_approach/train.py --val_fold 1 --tag fold1 --gpu 1 > logs/fold1.log 2>&1 &
python dfm_approach/train.py --val_fold 2 --tag fold2 --gpu 2 > logs/fold2.log 2>&1 &
wait  # 모든 백그라운드 작업 완료 대기
```

### 5-6. Multi-GPU + 여러 fold 동시 학습 예시

GPU 8개로 5 fold를 2-GPU DataParallel로 실행:

```bash
# GPU 0,1 → fold 0
python dfm_approach/train.py --val_fold 0 --tag fold0 --gpu 0 --num-gpus 2 &

# GPU 2,3 → fold 1
python dfm_approach/train.py --val_fold 1 --tag fold1 --gpu 2 --num-gpus 2 &

# GPU 4,5 → fold 2
python dfm_approach/train.py --val_fold 2 --tag fold2 --gpu 4 --num-gpus 2 &

# GPU 6,7 → fold 3
python dfm_approach/train.py --val_fold 3 --tag fold3 --gpu 6 --num-gpus 2 &

wait  # 4개 fold 병렬 완료 대기

# GPU 0,1 → fold 4
python dfm_approach/train.py --val_fold 4 --tag fold4 --gpu 0 --num-gpus 2
```

### 5-7. train.py 전체 옵션 요약

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--config` | `dfm_approach/config_dfm.txt` | config 파일 경로 |
| `--phase` | `0` (=전부) | 실행할 Phase (0=all, 1=FNO, 2=CAE, 3=CFM) |
| `--val_fold` | config 값 | Leave-one-out fold 인덱스 |
| `--gpu` | config의 `gpu_ids` | 시작 GPU 인덱스 |
| `--num-gpus` | `1` | DataParallel에 사용할 GPU 수 |
| `--tag` | None | 체크포인트/로그 파일명 접미사 |
| `--resume` | False | 기존 체크포인트에서 재개 |

---

## 6. GPU 설정 옵션 상세

### 단일 GPU

```bash
# config.txt의 gpu_ids 사용 (기본)
python dfm_approach/train.py

# GPU 3번 지정
python dfm_approach/train.py --gpu 3

# CPU 사용 (config에서 gpu_ids=-1로 설정)
```

### Multi-GPU (DataParallel)

```bash
# GPU 0,1 (2개)
python dfm_approach/train.py --gpu 0 --num-gpus 2

# GPU 2,3,4,5 (4개)
python dfm_approach/train.py --gpu 2 --num-gpus 4

# 전체 GPU 사용 (8개)
python dfm_approach/train.py --gpu 0 --num-gpus 8
```

**DataParallel 참고사항:**
- `--num-gpus 1`이면 DataParallel을 사용하지 않음 (기본 동작)
- 배치를 GPU 수로 나누어 병렬 처리 → 유효 배치 크기 = `batch_size × num_gpus`
- 메모리 효율: 각 GPU는 `batch_size / num_gpus` 만큼의 메모리 사용
- 체크포인트는 항상 단일 모델로 저장되므로, 이후 단일 GPU에서도 로드 가능

### Multi-GPU (Shell Script fold 분배)

```bash
# 10 fold를 8 GPU에 분배 (각 GPU에서 1~2 fold 실행)
bash dfm_approach/run_pipeline.sh --gpus 8

# 10 fold를 4 GPU에 분배 (각 GPU에서 2~3 fold 순차 실행)
bash dfm_approach/run_pipeline.sh --gpus 4
```

### 어떤 방식을 선택해야 하나?

| 상황 | 권장 방식 |
|------|----------|
| GPU 8개, 전체 fold 학습 | `run_pipeline.sh --gpus 8` |
| GPU 4개, 전체 fold 학습 | `run_pipeline.sh --gpus 4` |
| 특정 fold 1개만 디버깅 | `train.py --val_fold 0 --gpu 0` |
| GPU 2개로 1 fold 빠르게 | `train.py --val_fold 0 --gpu 0 --num-gpus 2` |
| GPU 8개, 4 fold씩 2-GPU DP | 수동 백그라운드 실행 (5-6 참조) |

---

## 7. 평가 및 샘플링

### 평가

```bash
# 단일 fold 평가
python dfm_approach/evaluate.py \
    --config dfm_approach/config_dfm.txt \
    --fold 0 \
    --k 50 \
    --gpu 0

# fold별 체크포인트 사용 시
python dfm_approach/evaluate.py \
    --fold 0 --tag fold0 --gpu 0

# 모든 fold 평가 (순차)
python dfm_approach/evaluate.py --all-folds --k 50 --gpu 0

# Shell Script로 모든 fold 병렬 평가
bash dfm_approach/run_evaluate_all.sh --gpus 8 --k 50
```

### 샘플링

```bash
# 단일 디자인에서 300개 샘플 생성
python dfm_approach/sample.py \
    --config dfm_approach/config_dfm.txt \
    --design /path/to/design.png \
    --num-samples 300 \
    --gpu 0

# 디렉토리 내 모든 디자인
python dfm_approach/sample.py \
    --design-dir /data2/AOI/B8_AOI_preprocessed/ \
    --num-samples 100 \
    --gpu 0

# 물리 단위(μm)로 역정규화
python dfm_approach/sample.py \
    --design /path/to/design.png \
    --denormalize \
    --gpu 0

# fold별 체크포인트 사용
python dfm_approach/sample.py \
    --design /path/to/design.png \
    --tag fold0 \
    --gpu 0
```

### evaluate.py 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--config` | `dfm_approach/config_dfm.txt` | config 파일 경로 |
| `--fold` | `0` | 평가할 fold 인덱스 |
| `--all-folds` | False | 모든 fold 순차 평가 |
| `--k` | `50` | fold당 생성할 샘플 수 |
| `--gpu` | config 값 | GPU 인덱스 |
| `--tag` | None | fold별 체크포인트 접미사 |

### sample.py 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--config` | `dfm_approach/config_dfm.txt` | config 파일 경로 |
| `--design` | None | 단일 디자인 PNG 경로 |
| `--design-dir` | None | 디자인 PNG 디렉토리 |
| `--num-samples` | config의 `num_gen_samples` | 생성할 샘플 수 |
| `--temperature` | `1.0` | 샘플링 온도 |
| `--denormalize` | False | 물리 단위(μm)로 역정규화 |
| `--gpu` | config 값 | GPU 인덱스 |
| `--tag` | None | fold별 체크포인트 접미사 |

---

## 8. 검증 스크립트

서버에 배포 후, 학습 전에 환경을 검증합니다:

```bash
# 기본 검증 (import, shape, GPU, 메모리)
python dfm_approach/verify.py --config dfm_approach/config_dfm.txt

# 학습 완료 후 결과 검증 포함
python dfm_approach/verify.py --config dfm_approach/config_dfm.txt --validate-results
```

검증 항목:
- 모든 모듈 import 가능 여부
- 모델 shape 일치 (forward pass)
- GPU 가용성 및 메모리 추정
- 잠재적 병목 지점 분석
- 튜닝 가이드라인 출력

---

## 9. 체크포인트 구조

### Phase 1 (FNO)
```python
{
    'epoch': int,
    'phase': 1,
    'model_state': OrderedDict,       # FNOMeanPredictor
    'optimizer_state': OrderedDict,
    'config': dict,
    'val_loss': float,
}
```

### Phase 2 (CAE)
```python
{
    'epoch': int,
    'phase': 2,
    'cond_enc_state': OrderedDict,    # ConditionEncoder
    'cae_state': OrderedDict,         # ResidualCAE
    'optimizer_state': OrderedDict,
    'config': dict,
    'val_loss': float,
}
```

### Phase 3 (OT-CFM)
```python
{
    'epoch': int,
    'phase': 3,
    'cfm_state': OrderedDict,         # OTCFM (velocity_net)
    'ema_state': dict,                 # EMA shadow weights
    'optimizer_state': OrderedDict,
    'config': dict,
    'val_loss': float,
    'fno_checkpoint': str,             # Phase 1 체크포인트 경로
    'cae_checkpoint': str,             # Phase 2 체크포인트 경로
}
```

---

## 10. 트러블슈팅

### Phase 2/3 실행 시 "checkpoint not found" 에러

Phase 2는 Phase 1의 FNO 체크포인트가, Phase 3는 Phase 1+2의 체크포인트가 필요합니다.
`--tag` 사용 시 evaluate/sample에도 같은 `--tag`를 전달해야 합니다.

```bash
# train 시 --tag fold0 사용
python dfm_approach/train.py --val_fold 0 --tag fold0 --gpu 0

# evaluate 시에도 동일한 --tag 필요
python dfm_approach/evaluate.py --fold 0 --tag fold0 --gpu 0
```

### GPU 메모리 부족 (OOM)

`config_dfm.txt`에서 배치 크기를 줄입니다:
```
fno_batch_size      8     # 기본 16 → 8
cae_batch_size      16    # 기본 32 → 16
cfm_batch_size      32    # 기본 64 → 32
```

또는 `--num-gpus`로 GPU를 추가하면 GPU당 배치 크기가 줄어듭니다.

### Shell script "permission denied"

```bash
chmod +x dfm_approach/run_pipeline.sh
chmod +x dfm_approach/run_all_folds.sh
chmod +x dfm_approach/run_evaluate_all.sh
```

### 로그 확인

```bash
# 실시간 로그 확인
tail -f outputs/train_dfm.log
tail -f outputs/train_dfm_fold0.log

# Shell Script 파이프라인 로그
tail -f outputs/dfm_pipeline_*.log

# fold별 로그
tail -f outputs/logs_dfm/fold0_phase1.log
```
