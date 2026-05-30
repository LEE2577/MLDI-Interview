#!/bin/bash
# Measure test-time latency and FLOPs vs. base model, verify ≤1.1x budget.
# Each model is measured in its own isolated Python process for a fair comparison.
#
# Usage:
#   SEED=1 bash run_measure.sh
#   BASE_MODEL=./models/Qwen3-VL-2B-Instruct MODEL_PATH=./outputs/.../merged bash run_measure.sh
#   SKIP_FLOPS=true SEED=1 bash run_measure.sh
set -euo pipefail

BASE_MODEL=${BASE_MODEL:-./models/Qwen3-VL-2B-Instruct}
SEED=${SEED:-1}
SKIP_FLOPS=${SKIP_FLOPS:-false}

# Auto-detect merged model for the given seed if MODEL_PATH is not set
if [ -z "${MODEL_PATH:-}" ]; then
    CANDIDATE="./outputs/textvqa_qwen3vl_lora_seed${SEED}/merged"
    if [ -d "${CANDIDATE}" ]; then
        MODEL_PATH="${CANDIDATE}"
        echo "[run_measure] Auto-detected merged model: ${MODEL_PATH}"
    fi
fi

EXTRA_ARGS=""
[ "${SKIP_FLOPS}" = "true" ] && EXTRA_ARGS="--skip_flops"

BASE_JSON=$(mktemp /tmp/efficiency_base_XXXXXX.json)
FT_JSON=$(mktemp /tmp/efficiency_ft_XXXXXX.json)
trap 'rm -f "${BASE_JSON}" "${FT_JSON}"' EXIT

echo "[run_measure] Step 1/3: measuring base model in isolated process ..."
python measure_efficiency.py \
    --model "${BASE_MODEL}" \
    --save "${BASE_JSON}" \
    ${EXTRA_ARGS}

if [ -n "${MODEL_PATH:-}" ]; then
    echo ""
    echo "[run_measure] Step 2/3: measuring fine-tuned model in isolated process ..."
    python measure_efficiency.py \
        --model "${MODEL_PATH}" \
        --save "${FT_JSON}" \
        ${EXTRA_ARGS}

    echo ""
    echo "[run_measure] Step 3/3: comparing results ..."
    python measure_efficiency.py --compare "${BASE_JSON}" "${FT_JSON}"
else
    echo "[run_measure] No MODEL_PATH set; base-only measurement complete."
    echo "[run_measure] Run with MODEL_PATH=<merged_model_dir> to compare."
fi
