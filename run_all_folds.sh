#!/usr/bin/env bash
# =============================================================
# run_all_folds.sh — Run leave-one-out training across multiple GPUs
#
# Distributes folds across N GPUs in parallel (round-robin).
# Number of folds is auto-detected from design_names in config.
# GPUs with >1 fold handle them sequentially.
# Stops immediately if any fold fails.
#
# Usage:
#   bash run_all_folds.sh                                # CVAE (default)
#   bash run_all_folds.sh --model ldm                    # LDM
#   bash run_all_folds.sh --model lfm                    # LFM
#   bash run_all_folds.sh --model cvae --gpus 4          # Use only 4 GPUs
#   bash run_all_folds.sh --model ldm --gpu-offset 4     # Start from GPU 4
# =============================================================

set -euo pipefail

MODEL="cvae"
NUM_GPUS=8
GPU_OFFSET=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)      MODEL="$2"; shift 2 ;;
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        --gpu-offset) GPU_OFFSET="$2"; shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Select config file based on model type
case $MODEL in
    cvae) CONFIG="config.txt" ;;
    ldm)  CONFIG="config_ldm.txt" ;;
    lfm)  CONFIG="config_lfm.txt" ;;
    *)    echo "Unknown model: $MODEL (choose cvae/ldm/lfm)"; exit 1 ;;
esac

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
echo "Auto-detected $NUM_FOLDS designs from $CONFIG"

LOG_DIR="outputs/logs_${MODEL}"
FAIL_FLAG="/tmp/run_all_folds_${MODEL}_$$_fail"
mkdir -p "$LOG_DIR"
rm -f "$FAIL_FLAG"

echo "============================================"
echo "  Model: ${MODEL^^}"
echo "  Config: $CONFIG"
echo "  GPUs: $GPU_OFFSET..$((GPU_OFFSET + NUM_GPUS - 1))"
echo "  Folds: $NUM_FOLDS"
echo "============================================"

# Function to train a single fold on a specific GPU
train_fold() {
    local fold=$1
    local gpu=$2
    local tag="fold${fold}"
    local logfile="${LOG_DIR}/${tag}.log"

    # Build extra args for LDM/LFM: point to fold-specific CVAE checkpoint
    local extra_args=""
    if [ "$MODEL" = "ldm" ] || [ "$MODEL" = "lfm" ]; then
        local cvae_ckpt="outputs/cvae_pcb_${tag}.pth"
        if [ ! -f "$cvae_ckpt" ]; then
            echo "[GPU $gpu] ERROR: CVAE checkpoint not found: $cvae_ckpt"
            touch "$FAIL_FLAG"
            return 1
        fi
        extra_args="--cvae-checkpoint $cvae_ckpt"
    fi

    echo "[GPU $gpu] Starting ${MODEL^^} fold $fold → $logfile"
    if ! python train.py \
        --config "$CONFIG" \
        --val_fold "$fold" \
        --gpu "$gpu" \
        --tag "$tag" \
        $extra_args \
        > "$logfile" 2>&1; then
        echo "[GPU $gpu] ERROR: ${MODEL^^} fold $fold failed! See $logfile"
        touch "$FAIL_FLAG"
        return 1
    fi

    echo "[GPU $gpu] ${MODEL^^} fold $fold finished successfully"
}

# Distribute folds across GPUs: round-robin assignment
declare -a gpu_assignments
for fold in $(seq 0 $((NUM_FOLDS - 1))); do
    gpu_assignments[$fold]=$(( (fold % NUM_GPUS) + GPU_OFFSET ))
done

# Group folds by GPU and run each GPU's folds sequentially,
# but all GPUs run in parallel.
# Each GPU subshell stops on first error.
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
            # Stop this GPU's queue if a failure was detected anywhere
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
echo "(Monitor progress: tail -f ${LOG_DIR}/fold*.log)"
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
    echo "  All ${MODEL^^} folds completed successfully!"
else
    echo "  ERROR: $fail GPU group(s) had failures."
    echo "  Check logs in $LOG_DIR/"
    exit 1
fi
echo "============================================"
