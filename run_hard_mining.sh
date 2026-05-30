#!/bin/bash
# Score training samples by base-model loss and export a hard subset.
# Run once per seed after run_prepare.sh, before run_train.sh.
#
# Usage:
#   SEED=1 bash run_hard_mining.sh
#
# Options (override via env):
#   HARD_RATIO   fraction of hardest samples to keep  (default: 0.5)
#   BATCH_SIZE   inference batch size                 (default: 4)

set -e
cd "$(dirname "$0")"

SEED=${SEED:-1}
HARD_RATIO=${HARD_RATIO:-0.5}
BATCH_SIZE=${BATCH_SIZE:-4}

echo "[run_hard_mining] SEED=${SEED} HARD_RATIO=${HARD_RATIO} BATCH_SIZE=${BATCH_SIZE}"

SEED=${SEED} python prepare_hard_mining.py \
    --config configs/vlm_textvqa_lora.yaml \
    --batch_size ${BATCH_SIZE} \
    --hard_ratio ${HARD_RATIO}
