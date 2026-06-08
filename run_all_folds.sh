#!/usr/bin/env bash
# =============================================================
# run_all_folds.sh — Run leave-one-out training across multiple GPUs
#
# Distributes 10 folds across N GPUs in parallel (round-robin).
# GPUs with >1 fold handle them sequentially.
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

NUM_FOLDS=10
LOG_DIR="outputs/logs_${MODEL}"
mkdir -p "$LOG_DIR"

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
            echo "[GPU $gpu] WARNING: CVAE checkpoint not found: $cvae_ckpt — skipping fold $fold"
            return 1
        fi
        extra_args="--cvae-checkpoint $cvae_ckpt"
    fi

    echo "[GPU $gpu] Starting ${MODEL^^} fold $fold → $logfile"
    python train.py \
        --config "$CONFIG" \
        --val_fold "$fold" \
        --gpu "$gpu" \
        --tag "$tag" \
        $extra_args \
        > "$logfile" 2>&1

    echo "[GPU $gpu] ${MODEL^^} fold $fold finished (exit=$?)"
}

# Distribute folds across GPUs: round-robin assignment
declare -a gpu_assignments
for fold in $(seq 0 $((NUM_FOLDS - 1))); do
    gpu_assignments[$fold]=$(( (fold % NUM_GPUS) + GPU_OFFSET ))
done

# Group folds by GPU and run each GPU's folds sequentially,
# but all GPUs run in parallel
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

echo ""
echo "============================================"
if [ $fail -eq 0 ]; then
    echo "  All ${MODEL^^} folds completed successfully!"
else
    echo "  WARNING: $fail GPU group(s) had errors."
    echo "  Check logs in $LOG_DIR/"
fi
echo "============================================"
