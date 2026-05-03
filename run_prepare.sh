#!/bin/bash
set -euo pipefail

export CONFIG=${CONFIG:-configs/vlm_textvqa_lora.yaml}
export SEED=${SEED:-1}

python prepare_textvqa.py --config "${CONFIG}"
