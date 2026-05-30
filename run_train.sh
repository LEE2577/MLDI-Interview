#!/bin/bash
set -euo pipefail

export WANDB_DISABLED=true
export CONFIG=${CONFIG:-configs/vlm_textvqa_lora.yaml}
export SEED=${SEED:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_GPUS=${NUM_GPUS:-$(python -c "import torch; print(torch.cuda.device_count())")}

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

if [ "${NUM_GPUS}" -gt 1 ]; then
  accelerate launch --num_processes "${NUM_GPUS}" --multi_gpu --mixed_precision fp16 train_textvqa_qwen3vl.py --config "${CONFIG}"
else
  accelerate launch --num_processes 1 --mixed_precision fp16 train_textvqa_qwen3vl.py --config "${CONFIG}"
fi
