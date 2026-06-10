#!/usr/bin/env bash
# =============================================================
# run_evaluate_all.sh — DF²M evaluation + sampling across GPUs
#
# Runs evaluate.py and sample.py for all folds in parallel.
#
# Usage:
#   bash dfm_approach/run_evaluate_all.sh --gpus 8
#   bash dfm_approach/run_evaluate_all.sh --gpus 4 --config config_dfm.txt
# =============================================================

set -euo pipefail

# cd to the directory containing this script (dfm_approach/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_GPUS=8
GPU_OFFSET=0
CONFIG="config_dfm.txt"
K=50

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        --gpu-offset) GPU_OFFSET="$2"; shift 2 ;;
        --config)     CONFIG="$2"; shift 2 ;;
        --k)          K="$2"; shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Auto-detect number of folds
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

LOG_DIR="outputs/logs_dfm"
FAIL_FLAG="/tmp/dfm_eval_$$_fail"
mkdir -p "$LOG_DIR"
rm -f "$FAIL_FLAG"

echo "============================================"
echo "  DF²M Evaluate + Sample"
echo "  Config: $CONFIG"
echo "  GPUs: $GPU_OFFSET..$((GPU_OFFSET + NUM_GPUS - 1))"
echo "  Folds: $NUM_FOLDS | K=$K"
echo "============================================"

eval_fold() {
    local fold=$1
    local gpu=$2
    local tag="fold${fold}"
    local logfile="${LOG_DIR}/${tag}_eval.log"

    echo "[GPU $gpu] Evaluating fold $fold → $logfile"

    # Evaluate: point to fold-specific checkpoints
    if ! CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" python evaluate.py \
        --config "$CONFIG" \
        --fold "$fold" \
        --k "$K" \
        --gpu 0 \
        --tag "$tag" \
        > "$logfile" 2>&1; then
        echo "[GPU $gpu] ERROR: Evaluation fold $fold failed! See $logfile"
        touch "$FAIL_FLAG"
        return 1
    fi

    echo "[GPU $gpu] Evaluation fold $fold finished"
}

sample_fold() {
    local fold=$1
    local gpu=$2
    local tag="fold${fold}"
    local logfile="${LOG_DIR}/${tag}_sample.log"

    echo "[GPU $gpu] Sampling fold $fold → $logfile"

    if ! CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" python sample.py \
        --config "$CONFIG" \
        --num-samples "$K" \
        --gpu 0 \
        --tag "$tag" \
        --denormalize \
        > "$logfile" 2>&1; then
        echo "[GPU $gpu] ERROR: Sampling fold $fold failed! See $logfile"
        touch "$FAIL_FLAG"
        return 1
    fi

    echo "[GPU $gpu] Sampling fold $fold finished"
}

# Round-robin fold assignment
declare -a gpu_assignments
for fold in $(seq 0 $((NUM_FOLDS - 1))); do
    gpu_assignments[$fold]=$(( (fold % NUM_GPUS) + GPU_OFFSET ))
done

pids=()
for gpu_slot in $(seq 0 $((NUM_GPUS - 1))); do
    gpu=$((gpu_slot + GPU_OFFSET))

    folds_for_gpu=()
    for fold in $(seq 0 $((NUM_FOLDS - 1))); do
        if [ "${gpu_assignments[$fold]}" -eq "$gpu" ]; then
            folds_for_gpu+=("$fold")
        fi
    done

    if [ ${#folds_for_gpu[@]} -eq 0 ]; then
        continue
    fi

    (
        for fold in "${folds_for_gpu[@]}"; do
            if [ -f "$FAIL_FLAG" ]; then
                echo "[GPU $gpu] Skipping fold $fold (earlier failure detected)"
                exit 1
            fi
            eval_fold "$fold" "$gpu"
        done
    ) &
    pids+=($!)
    echo "Launched GPU $gpu: eval folds [${folds_for_gpu[*]}]"
done

echo ""
echo "All GPUs launched. Waiting..."

fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        fail=$((fail + 1))
    fi
done

rm -f "$FAIL_FLAG"

if [ $fail -gt 0 ]; then
    echo ""
    echo "============================================"
    echo "  ERROR: $fail GPU group(s) had evaluation failures."
    echo "  Check logs in $LOG_DIR/"
    echo "============================================"
    exit 1
fi
echo "Evaluation phase complete."

# ==============================================================
# Phase B: Sample (all folds, parallel across GPUs)
# ==============================================================
echo ""
echo "--- Sampling phase ($NUM_FOLDS folds) ---"

pids=()
for gpu_slot in $(seq 0 $((NUM_GPUS - 1))); do
    gpu=$((gpu_slot + GPU_OFFSET))

    folds_for_gpu=()
    for fold in $(seq 0 $((NUM_FOLDS - 1))); do
        if [ "${gpu_assignments[$fold]}" -eq "$gpu" ]; then
            folds_for_gpu+=("$fold")
        fi
    done

    if [ ${#folds_for_gpu[@]} -eq 0 ]; then
        continue
    fi

    (
        for fold in "${folds_for_gpu[@]}"; do
            if [ -f "$FAIL_FLAG" ]; then
                echo "[GPU $gpu] Skipping sample fold $fold (earlier failure detected)"
                exit 1
            fi
            sample_fold "$fold" "$gpu"
        done
    ) &
    pids+=($!)
    echo "Launched GPU $gpu: sample folds [${folds_for_gpu[*]}]"
done

echo ""
echo "All GPUs launched for sampling. Waiting..."

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
    echo "  DF²M Evaluate + Sample — All folds completed!"
    echo "  Eval logs:   $LOG_DIR/*_eval.log"
    echo "  Sample logs: $LOG_DIR/*_sample.log"
else
    echo "  ERROR: $fail GPU group(s) had sampling failures."
    echo "  Check logs in $LOG_DIR/"
    exit 1
fi
echo "============================================"
