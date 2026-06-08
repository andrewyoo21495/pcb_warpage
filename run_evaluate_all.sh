#!/usr/bin/env bash
# =============================================================
# run_evaluate_all.sh — Batch evaluate + sample + visualize
#
# For each model (cvae/ldm/lfm), runs all 10 folds in parallel
# across available GPUs.
#
# Usage:
#   bash run_evaluate_all.sh                         # all 3 models
#   bash run_evaluate_all.sh --model cvae            # CVAE only
#   bash run_evaluate_all.sh --gpus 4                # use 4 GPUs
#   bash run_evaluate_all.sh --skip-sample           # evaluate only
#   bash run_evaluate_all.sh --skip-visualize        # no visualize_eval
#   bash run_evaluate_all.sh --k 50                  # 50 samples per design
# =============================================================

set -euo pipefail

MODELS="cvae ldm lfm"
NUM_GPUS=8
NUM_FOLDS=10
K_SAMPLES=""
SKIP_SAMPLE=false
SKIP_VISUALIZE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)          MODELS="$2"; shift 2 ;;
        --gpus)           NUM_GPUS="$2"; shift 2 ;;
        --k)              K_SAMPLES="$2"; shift 2 ;;
        --skip-sample)    SKIP_SAMPLE=true; shift ;;
        --skip-visualize) SKIP_VISUALIZE=true; shift ;;
        *)                echo "Unknown arg: $1"; exit 1 ;;
    esac
done

get_config() {
    case $1 in
        cvae) echo "config.txt" ;;
        ldm)  echo "config_ldm.txt" ;;
        lfm)  echo "config_lfm.txt" ;;
    esac
}

echo "============================================"
echo "  Models: $MODELS"
echo "  GPUs: $NUM_GPUS"
echo "  K samples: ${K_SAMPLES:-from config}"
echo "============================================"

for model in $MODELS; do
    conf=$(get_config "$model")
    log_dir="outputs/logs_${model}"
    mkdir -p "$log_dir"

    echo ""
    echo "========== ${model^^} =========="

    # --- Phase A: Evaluate ---
    echo "[${model^^}] Running evaluate.py (10 folds)..."
    pids=()
    for fold in $(seq 0 $((NUM_FOLDS - 1))); do
        gpu=$((fold % NUM_GPUS))
        tag="fold${fold}"
        ckpt="outputs/${model}_pcb_${tag}.pth"

        if [ ! -f "$ckpt" ]; then
            echo "  Fold $fold: checkpoint not found ($ckpt), skipping."
            continue
        fi

        k_arg=""
        if [ -n "$K_SAMPLES" ]; then k_arg="--k $K_SAMPLES"; fi

        python evaluate.py \
            --config "$conf" \
            --fold "$fold" \
            --checkpoint "$ckpt" \
            --gpu "$gpu" \
            $k_arg \
            > "${log_dir}/eval_${tag}.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || true; done
    echo "[${model^^}] Evaluate done."

    # --- Phase B: Sample ---
    if [ "$SKIP_SAMPLE" = false ]; then
        echo "[${model^^}] Running sample.py (10 folds)..."
        pids=()
        for fold in $(seq 0 $((NUM_FOLDS - 1))); do
            gpu=$((fold % NUM_GPUS))
            tag="fold${fold}"
            ckpt="outputs/${model}_pcb_${tag}.pth"

            if [ ! -f "$ckpt" ]; then continue; fi

            k_arg=""
            if [ -n "$K_SAMPLES" ]; then k_arg="--k $K_SAMPLES"; fi

            python sample.py \
                --config "$conf" \
                --design-dir data/design \
                --checkpoint "$ckpt" \
                --save "outputs/samples_${model}_${tag}" \
                --gpu "$gpu" \
                --denormalize \
                $k_arg \
                > "${log_dir}/sample_${tag}.log" 2>&1 &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do wait "$pid" || true; done
        echo "[${model^^}] Sample done."
    fi

    # --- Phase C: Visualize ---
    if [ "$SKIP_VISUALIZE" = false ]; then
        echo "[${model^^}] Running visualize_eval.py (10 folds)..."
        pids=()
        for fold in $(seq 0 $((NUM_FOLDS - 1))); do
            gpu=$((fold % NUM_GPUS))
            tag="fold${fold}"
            ckpt="outputs/${model}_pcb_${tag}.pth"

            if [ ! -f "$ckpt" ]; then continue; fi

            k_arg=""
            if [ -n "$K_SAMPLES" ]; then k_arg="--k $K_SAMPLES"; fi

            python visualize_eval.py \
                --config "$conf" \
                --fold "$fold" \
                --checkpoint "$ckpt" \
                --save "outputs/vis_${model}_${tag}" \
                --gpu "$gpu" \
                $k_arg \
                > "${log_dir}/vis_${tag}.log" 2>&1 &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do wait "$pid" || true; done
        echo "[${model^^}] Visualize done."
    fi
done

echo ""
echo "============================================"
echo "  All evaluation complete!"
echo "  Evaluate logs : outputs/logs_*/eval_fold*.log"
echo "  Sample outputs: outputs/samples_*/"
echo "  Visualizations: outputs/vis_*/"
echo "============================================"
