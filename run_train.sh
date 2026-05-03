#!/bin/bash
set -euo pipefail

export WANDB_DISABLED=true
export CONFIG=${CONFIG:-configs/vlm_textvqa_lora.yaml}
export SEED=${SEED:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

python - <<'PY'
import os
import sys
import yaml

config_path = os.environ["CONFIG"]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
seed = int(os.environ.get("SEED", cfg.get("seed", 1)))
prepared = os.environ.get("PREPARED_DATA_DIR", cfg["prepared_data_dir"]).format(seed=seed)
if not os.path.isdir(prepared):
    print(f"[ERROR] Prepared dataset not found: {prepared}", file=sys.stderr)
    print(f"[ERROR] Run first: SEED={seed} CONFIG={config_path} bash run_prepare.sh", file=sys.stderr)
    sys.exit(1)
PY

accelerate launch --num_processes 2 --multi_gpu --mixed_precision fp16 train_textvqa_qwen3vl.py --config "${CONFIG}"
