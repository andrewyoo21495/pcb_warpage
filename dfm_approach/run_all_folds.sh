#!/usr/bin/env bash
# =============================================================
# run_all_folds.sh — DF²M leave-one-out training across GPUs
#
# Distributes folds across N GPUs in parallel (round-robin).
# Each GPU runs its assigned folds sequentially.
# Stops immediately if any fold fails.
#
# Usage:
#   bash dfm_approach/run_all_folds.sh --phase 1                # Phase 1 (FNO), 8 GPUs
#   bash dfm_approach/run_all_folds.sh --phase 2 --gpus 4       # Phase 2 (CAE), 4 GPUs
#   bash dfm_approach/run_all_folds.sh --phase 3 --gpus 4 --gpu-offset 4
#   bash dfm_approach/run_all_folds.sh --phase 0 --gpus 2       # All phases, 2 GPUs
# =============================================================

set -euo pipefail

PHASE=0
NUM_GPUS=8
GPU_OFFSET=0
CONFIG="dfm_approach/config_dfm.txt"

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)      PHASE="$2"; shift 2 ;;
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        --gpu-offset) GPU_OFFSET="$2"; shift 2 ;;
        --config)     CONFIG="$2"; shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Auto-detect number of folds from design_names in config
NUM_FOLDS=$(python -c "
import sys, io
sys.stdout = io.StringIO()
from utils.load_config import load_config
c = load_config('$CONFIG')
sys.stdout = sys.__stdout__
names = c.get('design_names', [])
if isinstance(names, list):
    print(len(names))
else:
    print(len([n.strip() for n in str(names).split(',')]))
")

PHASE_LABEL="all"
case $PHASE in
    0) PHASE_LABEL="all (1→2→3)" ;;
    1) PHASE_LABEL="Phase 1 (FNO)" ;;
    2) PHASE_LABEL="Phase 2 (CAE)" ;;
    3) PHASE_LABEL="Phase 3 (OT-CFM)" ;;
esac

LOG_DIR="outputs/logs_dfm"
FAIL_FLAG="/tmp/dfm_run_all_folds_$$_fail"
mkdir -p "$LOG_DIR"
rm -f "$FAIL_FLAG"

echo "============================================"
echo "  DF²M Training: $PHASE_LABEL"
echo "  Config: $CONFIG"
echo "  GPUs: $GPU_OFFSET..$((GPU_OFFSET + NUM_GPUS - 1))"
echo "  Folds: $NUM_FOLDS"
echo "============================================"

# Function to train a single fold on a specific GPU
train_fold() {
    local fold=$1
    local gpu=$2
    local tag="fold${fold}"
    local logfile="${LOG_DIR}/${tag}_phase${PHASE}.log"

    echo "[GPU $gpu] Starting DF²M phase=${PHASE} fold $fold → $logfile"
    if ! python dfm_approach/train.py \
        --config "$CONFIG" \
        --val_fold "$fold" \
        --gpu "$gpu" \
        --tag "$tag" \
        --phase "$PHASE" \
        > "$logfile" 2>&1; then
        echo "[GPU $gpu] ERROR: DF²M fold $fold phase $PHASE failed! See $logfile"
        touch "$FAIL_FLAG"
        return 1
    fi

    echo "[GPU $gpu] DF²M fold $fold phase $PHASE finished successfully"
}

# Distribute folds across GPUs: round-robin assignment
declare -a gpu_assignments
for fold in $(seq 0 $((NUM_FOLDS - 1))); do
    gpu_assignments[$fold]=$(( (fold % NUM_GPUS) + GPU_OFFSET ))
done

# Group folds by GPU and run each GPU's folds sequentially,
# but all GPUs run in parallel.
pids=()
for gpu_slot in $(seq 0 $((NUM_GPUS - 1))); do
    gpu=$((gpu_slot + GPU_OFFSET))

    # Collect folds assigned to this GPU
    folds_for_gpu=()
    for fold in $(seq 0 $((NUM_FOLDS - 1))); do
        if [ "${gpu_assignments[$fold]}" -eq "$gpu" ]; then
            folds_for_gpu+=("$fold")
        fi
    done

    if [ ${#folds_for_gpu[@]} -eq 0 ]; then
        continue
    fi

    # Launch a background subshell for this GPU
    (
        for fold in "${folds_for_gpu[@]}"; do
            if [ -f "$FAIL_FLAG" ]; then
                echo "[GPU $gpu] Skipping fold $fold (earlier failure detected)"
                exit 1
            fi
            train_fold "$fold" "$gpu"
        done
    ) &
    pids+=($!)
    echo "Launched GPU $gpu: folds [${folds_for_gpu[*]}]"
done

echo ""
echo "All GPUs launched. Waiting for completion..."
echo "(Monitor progress: tail -f ${LOG_DIR}/fold*_phase${PHASE}.log)"
echo ""

# Wait for all background processes
fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        fail=$((fail + 1))
    fi
done

rm -f "$FAIL_FLAG"

echo ""
echo "============================================"
if [ $fail -eq 0 ]; then
    echo "  DF²M $PHASE_LABEL — All folds completed successfully!"
else
    echo "  ERROR: $fail GPU group(s) had failures."
    echo "  Check logs in $LOG_DIR/"
    exit 1
fi
echo "============================================"
