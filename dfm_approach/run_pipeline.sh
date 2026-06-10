#!/usr/bin/env bash
# =============================================================
# run_pipeline.sh — DF²M Full Training Pipeline
#
# 3-phase sequential training, each phase parallelized across GPUs:
#   Phase 1: FNO Mean Predictor       (all folds in parallel)
#   Phase 2: Residual CAE             (all folds in parallel)
#   Phase 3: OT-CFM Velocity Network  (all folds in parallel)
#   Phase 4: Evaluate + Sample        (all folds in parallel)
#
# Usage:
#   bash dfm_approach/run_pipeline.sh                    # all phases, 8 GPUs
#   bash dfm_approach/run_pipeline.sh --gpus 4           # all phases, 4 GPUs
#   bash dfm_approach/run_pipeline.sh --skip-to 2        # skip phase 1
#   bash dfm_approach/run_pipeline.sh --phase-only 3     # run only phase 3
#   bash dfm_approach/run_pipeline.sh --skip-eval        # skip evaluation
# =============================================================

set -euo pipefail

NUM_GPUS=8
GPU_OFFSET=0
SKIP_TO=1
PHASE_ONLY=0
SKIP_EVAL=false
CONFIG="dfm_approach/config_dfm.txt"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        --gpu-offset) GPU_OFFSET="$2"; shift 2 ;;
        --skip-to)    SKIP_TO="$2"; shift 2 ;;
        --phase-only) PHASE_ONLY="$2"; shift 2 ;;
        --skip-eval)  SKIP_EVAL=true; shift ;;
        --config)     CONFIG="$2"; shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="outputs/dfm_pipeline_${TIMESTAMP}.log"
mkdir -p outputs

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"
}

die() {
    log "FATAL: $*"
    log "Pipeline aborted. Check logs above."
    exit 1
}

log "============================================"
log "  DF²M Training Pipeline"
log "  GPUs: $GPU_OFFSET..$((GPU_OFFSET + NUM_GPUS - 1))"
log "  Config: $CONFIG"
log "  Pipeline log: $PIPELINE_LOG"
log "============================================"

# Determine which phases to run
if [ "$PHASE_ONLY" -gt 0 ]; then
    phases=("$PHASE_ONLY")
    log "Running ONLY Phase $PHASE_ONLY"
else
    phases=()
    for p in 1 2 3; do
        if [ "$p" -ge "$SKIP_TO" ]; then
            phases+=("$p")
        else
            log "Skipping Phase $p (--skip-to $SKIP_TO)"
        fi
    done
fi

# ==============================================================
# Phase 1: FNO Mean Predictor
# ==============================================================
if [[ " ${phases[*]} " =~ " 1 " ]]; then
    log ""
    log "========== Phase 1: FNO Mean Predictor (${NUM_GPUS} GPUs) =========="
    if ! bash dfm_approach/run_all_folds.sh \
        --phase 1 --gpus "$NUM_GPUS" --gpu-offset "$GPU_OFFSET" --config "$CONFIG" \
        2>&1 | tee -a "$PIPELINE_LOG"; then
        die "Phase 1 (FNO training) failed."
    fi
    log "Phase 1 complete."
fi

# ==============================================================
# Phase 2: Residual CAE + Condition Encoder
# ==============================================================
if [[ " ${phases[*]} " =~ " 2 " ]]; then
    log ""
    log "========== Phase 2: Residual CAE (${NUM_GPUS} GPUs) =========="
    if ! bash dfm_approach/run_all_folds.sh \
        --phase 2 --gpus "$NUM_GPUS" --gpu-offset "$GPU_OFFSET" --config "$CONFIG" \
        2>&1 | tee -a "$PIPELINE_LOG"; then
        die "Phase 2 (CAE training) failed."
    fi
    log "Phase 2 complete."
fi

# ==============================================================
# Phase 3: OT-CFM Velocity Network
# ==============================================================
if [[ " ${phases[*]} " =~ " 3 " ]]; then
    log ""
    log "========== Phase 3: OT-CFM (${NUM_GPUS} GPUs) =========="
    if ! bash dfm_approach/run_all_folds.sh \
        --phase 3 --gpus "$NUM_GPUS" --gpu-offset "$GPU_OFFSET" --config "$CONFIG" \
        2>&1 | tee -a "$PIPELINE_LOG"; then
        die "Phase 3 (OT-CFM training) failed."
    fi
    log "Phase 3 complete."
fi

# ==============================================================
# Phase 4: Evaluate + Sample (all folds)
# ==============================================================
if [ "$SKIP_EVAL" = false ]; then
    log ""
    log "========== Phase 4: Evaluate + Sample (${NUM_GPUS} GPUs) =========="
    if ! bash dfm_approach/run_evaluate_all.sh \
        --gpus "$NUM_GPUS" --gpu-offset "$GPU_OFFSET" --config "$CONFIG" \
        2>&1 | tee -a "$PIPELINE_LOG"; then
        die "Phase 4 (evaluation) failed."
    fi
    log "Phase 4 complete."
fi

log ""
log "============================================"
log "  DF²M Pipeline complete!"
log "  Logs:           outputs/logs_dfm/"
log "  Checkpoints:    outputs/dfm_*.pth"
log "  Samples:        outputs/samples_dfm/"
log "  Visualizations: outputs/vis_dfm/"
log "  Pipeline log:   $PIPELINE_LOG"
log "============================================"
