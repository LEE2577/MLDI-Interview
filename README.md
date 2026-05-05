# TextVQA Qwen3-VL LoRA Baseline

This repository contains a small VLM fine-tuning baseline for TextVQA. The current implementation fine-tunes `Qwen3-VL-2B-Instruct` with LoRA, but submissions may use any training structure or adaptation method.

**Deadline:** 3 days.

## Requirements

- Training must finish within 1 hour on 2 x 2080Ti.
- Test-time latency and FLOPs must be no more than 1.1x the original base model.
- The submitted model may use LoRA, full fine-tuning, adapters, prompt tuning, or another method, as long as the evaluation budget is respected.
- Final results should be reported over 3 seeds.

The final submission should include:

- `README.md`
- `run_prepare.sh`
- `run_train.sh`
- `eval_qwen.sh`
- Merge script if needed, for example `run_merge_lora.sh` and `merge_lora.py`
- Any required config and source files used by those scripts

## Files

- `configs/vlm_textvqa_lora.yaml`: training configuration.
- `prepare_textvqa.py`: prepares and caches TextVQA prompts.
- `train_textvqa_qwen3vl.py`: LoRA fine-tuning script.
- `run_prepare.sh`: data preparation entrypoint.
- `run_train.sh`: 2-GPU training entrypoint.
- `run_merge_lora.sh`: merges a LoRA adapter into the base model.
- `eval_qwen.sh`: TextVQA evaluation entrypoint based on `lmms-eval`.

## Data And Model

The default config uses:

- **Model:** `Qwen3-VL-2B-Instruct` (local path or Hugging Face)
- **Dataset:** TextVQA from Hugging Face (`lmms-lab/textvqa`). You can also point `data_path` in the config to local `*.parquet` files.

```yaml
model_path: Qwen/Qwen3-VL-2B-Instruct
data_path: lmms-lab/textvqa
```

Prepared data is saved under `data/prepared_textvqa_qwen3vl_seed{seed}`. Training outputs are saved under `outputs/textvqa_qwen3vl_lora_seed{seed}`.

The default prompt does not include dataset-provided OCR tokens.

## Base Model Performance

The original `Qwen3-VL-2B-Instruct` (without fine-tuning) achieves the following on `textvqa_val`:

| Model | exact_match |
|-------|-------------|
| Qwen3-VL-2B-Instruct | **69.84%** |

## Baseline (LoRA Fine-tuned) Performance

This LoRA fine-tuning baseline achieves the following on `textvqa_val` across 3 seeds:

| Seed | exact_match |
|------|-------------|
| 1    | 70.63%      |
| 2    | 70.73%      |
| 3    | 70.67%      |
| **Mean** | **70.68%** |

> This is a simple baseline. Students are expected to surpass this score. Achieving a comparable result with better code quality and innovative ideas is also acceptable.

## Setup

### Environment

You may use the pre-configured shared environment, or install dependencies yourself:

```bash
pip install -r requirements.txt
cd lmms-eval && pip install -e . && cd ..
```

> If you install your own environment, **include your `requirements.txt`** in the submission.

### Coding Style

You are free to use any workflow — including vibe coding tools like Claude Code, Cursor, GitHub Copilot, etc. — as long as the submitted code is clean, reproducible, and runs correctly.

## Quick Start

### 1. Prepare data

```bash
SEED=1 bash run_prepare.sh
```

### 2. Train

```bash
SEED=1 bash run_train.sh
```

For 3 seeds:

```bash
for seed in 1 2 3; do
  SEED=$seed bash run_prepare.sh
  SEED=$seed bash run_train.sh
done
```

Training is controlled by `max_steps` and `max_train_seconds` in `configs/vlm_textvqa_lora.yaml`.

### 3. Merge LoRA

```bash
SEED=1 bash run_merge_lora.sh
```

The merged model is saved to `outputs/textvqa_qwen3vl_lora_seed1/merged` by default.

### 4. Evaluate

Evaluate the merged model:

```bash
MODEL_PATH=./outputs/textvqa_qwen3vl_lora_seed1/merged bash eval_qwen.sh
```

Evaluate the base model:

```bash
MODEL_PATH=/storage/data/shiyd2023/LLM_model/Qwen3-VL-2B-Instruct bash eval_qwen.sh
```
