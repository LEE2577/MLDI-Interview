# TextVQA Qwen3-VL-2B Fine-tuning: Experiment Report

## Environment

| Item | Details |
|---|---|
| GPU | GeForce RTX 4060 Laptop (single GPU) |
| Reference baseline | 2× RTX 2080Ti (official) |
| Base model | Qwen3-VL-2B-Instruct |
| Metric | Exact Match (EM) + Inference Throughput |

---

## Pipeline

```bash
# Data preparation + hard mining
SEED=1 bash run_prepare.sh
SEED=1 HARD_RATIO=0.5 bash run_hard_mining.sh

# Train → merge → evaluate
SEED=1 bash run_train.sh
SEED=1 bash run_merge_lora.sh
MODEL_PATH=./outputs/textvqa_qwen3vl_lora_seed1/merged bash eval_qwen.sh

# Throughput benchmark
SEED=1 bash run_measure.sh

# Full 3-seed run
for seed in 1 2 3; do
  SEED=$seed bash run_prepare.sh
  SEED=$seed bash run_train.sh
  SEED=$seed bash run_merge_lora.sh
  MODEL_PATH=./outputs/textvqa_qwen3vl_lora_seed${seed}/merged bash eval_qwen.sh
done
```

---

## Model Configuration

| Item | Value |
|---|---|
| Trainable parameters | 17,432,576 |
| Total parameters | 2,144,964,608 |
| Trainable % | 0.8127% |

---

## Results

### Exact Match on textvqa_val

| Method | Exact Match | Δ vs Baseline |
|---|---|---|
| Base model (no fine-tuning) | 69.84% | — |
| **Official LoRA baseline** (2×2080Ti, mean of 3 seeds) | **70.68%** | — |
| Baseline LoRA (this repo, seed=1) | 70.51% | — |
| + DoRA + OCR tokens | 69.99% | −0.52% |
| + Vision encoder LoRA | 70.26% | −0.25% |
| + OCR deduplication | 70.52% | +0.01% |
| **MoE-LoRA (OCR=false)** ✅ | **70.66%** | **+0.15%** |

### Inference Throughput

| Method | Gen tokens | Elapsed (s) | Speed (tokens/s) |
|---|---|---|---|
| Baseline LoRA | 18,654 | 2,946.8 | 6.33 |
| OCR deduplication | 18,626 | 2,849.2 | **6.54** |
| MoE-LoRA | 19,263 | 2,975.4 | 6.47 |

---

## Ablation Study

### Exp 1 — DoRA + OCR tokens
Replaced LoRA with DoRA and injected dataset-provided OCR tokens into the prompt. EM dropped to **69.99%**. Despite DoRA's stronger expressive capacity, raw OCR tokens introduced noise (misaligned text, irrelevant recognitions), distracting the model rather than helping it.

### Exp 2 — Vision Encoder LoRA
Added LoRA adapters to the vision encoder in addition to the language model. EM was **70.26%**, slightly below baseline. Under the memory constraints of a single 4060 Laptop, the larger trainable parameter count prevented full convergence within the time budget.

### Exp 3 — OCR Deduplication
Applied deduplication to OCR token sequences before training. EM reached **70.52%**, marginally above baseline. More notably, shorter prompts reduced total inference time by ~100 s, pushing throughput to **6.54 tokens/s** (+3.3%), well within the ≤1.1× latency budget.

### Exp 4 — MoE-LoRA, OCR=false ✅ Best
Replaced standard LoRA with a Mixture-of-Experts LoRA. With OCR input disabled, the model relies entirely on its own visual understanding. This achieved the highest EM of **70.66%** — nearly matching the official dual-GPU baseline (70.68%) on a single mobile GPU. Inference speed (6.47 tokens/s) satisfies the latency constraint.

---

## Key Takeaways

**Model architecture:** MoE-LoRA improves parameter utilization without increasing inference cost, outperforming all other variants.

**Data quality > data quantity:** Naively appending OCR tokens hurts performance. Deduplication recovers accuracy *and* improves throughput by shortening prompts.

**Hardware efficiency:** All experiments were completed on a single RTX 4060 Laptop. The best submission matches the official 2×2080Ti baseline to within 0.02% EM.

---

## Submitted Files

| File | Description |
|---|---|
| `README.md` | This file |
| `run_prepare.sh` | Data preparation entrypoint |
| `run_hard_mining.sh` | Hard example mining (HARD_RATIO=0.5) |
| `run_train.sh` | Training entrypoint |
| `run_merge_lora.sh` | Merge LoRA adapter into base model |
| `run_measure.sh` | Inference throughput benchmark |
| `eval_qwen.sh` | TextVQA evaluation via lmms-eval |
| `merge_lora.py` | LoRA merge script |
| `configs/vlm_textvqa_lora.yaml` | Training configuration |
| `prepare_textvqa.py` | Data preparation script |
| `train_textvqa_qwen3vl.py` | Fine-tuning script |