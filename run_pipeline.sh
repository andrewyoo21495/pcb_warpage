#!/usr/bin/env bash
# =============================================================
# run_pipeline.sh — Full training pipeline: CVAE → LDM + LFM
#
# Phase 1: Train CVAE on all folds (8 GPUs in parallel)
# Phase 2: Train LDM + LFM simultaneously
#            - LDM on GPUs 0..3, LFM on GPUs 4..7
# Phase 3: Evaluate + Sample + Visualize all models
#
# Stops immediately on any error.
#
# Usage:
#   bash run_pipeline.sh                # full pipeline
#   bash run_pipeline.sh --skip-cvae    # skip phase 1 (CVAE already trained)
#   bash run_pipeline.sh --gpus 4       # use only 4 GPUs
#   bash run_pipeline.sh --cvae-only    # train only CVAE
# =============================================================

set -euo pipefail

NUM_GPUS=8
SKIP_CVAE=false
CVAE_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        --skip-cvae)  SKIP_CVAE=true; shift ;;
        --cvae-only)  CVAE_ONLY=true; shift ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="outputs/pipeline_${TIMESTAMP}.log"
mkdir -p outputs

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"
}

die() {
    log "FATAL: $*"
    log "Pipeline aborted. Check logs above."
    exit 1
}

# ==============================================================
# Phase 1: CVAE training (all folds)
# ==============================================================
if [ "$SKIP_CVAE" = false ]; then
    log "========== Phase 1: CVAE Training (${NUM_GPUS} GPUs) =========="
    if ! bash run_all_folds.sh --model cvae --gpus "$NUM_GPUS" 2>&1 | tee -a "$PIPELINE_LOG"; then
        die "Phase 1 (CVAE training) failed."
    fi
    log "Phase 1 complete."
else
    log "Skipping Phase 1 (CVAE). Using existing checkpoints."
fi

if [ "$CVAE_ONLY" = true ]; then
    log "CVAE-only mode. Stopping after Phase 1."
    exit 0
fi

# ==============================================================
# Phase 2: LDM + LFM training in parallel
#
# Split GPUs evenly: first half → LDM, second half → LFM
# Each half uses run_all_folds.sh with --gpu-offset
# ==============================================================
log "========== Phase 2: LDM + LFM Training (parallel) =========="

HALF=$((NUM_GPUS / 2))
if [ $HALF -lt 1 ]; then HALF=1; fi

log "LDM on GPUs 0..$((HALF-1)), LFM on GPUs ${HALF}..$((NUM_GPUS-1))"

bash run_all_folds.sh --model ldm --gpus "$HALF" --gpu-offset 0 \
    2>&1 | tee -a "$PIPELINE_LOG" &
LDM_PID=$!

bash run_all_folds.sh --model lfm --gpus "$HALF" --gpu-offset "$HALF" \
    2>&1 | tee -a "$PIPELINE_LOG" &
LFM_PID=$!

fail=0
if ! wait "$LDM_PID"; then fail=$((fail + 1)); log "ERROR: LDM training failed"; fi
if ! wait "$LFM_PID"; then fail=$((fail + 1)); log "ERROR: LFM training failed"; fi

if [ $fail -gt 0 ]; then
    die "Phase 2 failed ($fail model(s) had errors)."
fi
log "Phase 2 complete."

# ==============================================================
# Phase 3: Evaluate + Sample + Visualize (all models)
# ==============================================================
log "========== Phase 3: Evaluate + Sample + Visualize =========="
if ! bash run_evaluate_all.sh --gpus "$NUM_GPUS" 2>&1 | tee -a "$PIPELINE_LOG"; then
    die "Phase 3 (evaluation) failed."
fi

log ""
log "============================================"
log "  Pipeline complete!"
log "  Logs:           outputs/logs_*/"
log "  Samples:        outputs/samples_*/"
log "  Visualizations: outputs/vis_*/"
log "  Pipeline log:   $PIPELINE_LOG"
log "============================================"
