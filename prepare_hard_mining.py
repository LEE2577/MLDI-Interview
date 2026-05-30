#!/usr/bin/env python
"""Hard example mining: score training samples by per-sample loss on the base model.

Saves difficulty_scores.json and a hard_subset/ dataset inside prepared_data_dir.
Run once before training; then set use_hard_mining: true in the config.
"""
import argparse
import json
import os

import torch
import yaml
from datasets import load_from_disk
from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

from train_textvqa_qwen3vl import TextVQADataset, collate_fn

DEFAULT_CONFIG = "configs/vlm_textvqa_lora.yaml"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(os.getenv("SEED", cfg.get("seed", 1)))
    cfg["seed"] = seed
    cfg["prepared_data_dir"] = os.getenv("PREPARED_DATA_DIR", cfg["prepared_data_dir"]).format(seed=seed)
    return cfg


@torch.no_grad()
def compute_per_sample_losses(model, dataset, processor, batch_size):
    model.eval()
    losses = []
    n = len(dataset)
    for start in range(0, n, batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, n))]
        batch = collate_fn(items, processor)
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        labels = batch.pop("labels")
        outputs = model(**batch)
        logits = outputs.logits  # [B, T, V]

        for b in range(logits.shape[0]):
            # Causal LM: logits[t] predicts labels[t+1], so shift by 1
            shifted_logits = logits[b, :-1, :]   # [T-1, V]
            shifted_labels = labels[b, 1:]        # [T-1]
            valid = shifted_labels != -100
            if not valid.any():
                losses.append(0.0)
                continue
            loss = torch.nn.functional.cross_entropy(
                shifted_logits[valid].float(), shifted_labels[valid]
            )
            losses.append(loss.item())

        if (start // batch_size) % 50 == 0:
            done = min(start + batch_size, n)
            recent = losses[-batch_size:]
            print(f"  [{done}/{n}] recent avg loss={sum(recent)/len(recent):.4f}")

    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hard_ratio", type=float, default=0.5,
                        help="Top fraction of samples by loss to keep as hard subset")
    parser.add_argument("--min_loss_pct", type=float, default=0.0,
                        help="Exclude samples below this loss percentile (removes trivial noise at extreme end)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    prepared_dir = cfg["prepared_data_dir"]
    if not os.path.isdir(prepared_dir):
        raise FileNotFoundError(f"Prepared dataset not found: {prepared_dir}. Run run_prepare.sh first.")

    processor = AutoProcessor.from_pretrained(
        cfg["model_path"], trust_remote_code=True,
        max_pixels=int(cfg["max_pixels"]), min_pixels=int(cfg["min_pixels"]),
    )
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model_path"], trust_remote_code=True,
        dtype=torch.float16,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        low_cpu_mem_usage=True,
    ).cuda()
    model.config.use_cache = False

    raw_ds = load_from_disk(prepared_dir)
    dataset = TextVQADataset(raw_ds, processor, cfg)
    print(f"[INFO] Scoring {len(dataset)} samples with base model (batch_size={args.batch_size})...")

    losses = compute_per_sample_losses(model, dataset, processor, args.batch_size)

    scores_path = os.path.join(prepared_dir, "difficulty_scores.json")
    with open(scores_path, "w") as f:
        json.dump({"losses": losses, "hard_ratio": args.hard_ratio}, f)
    print(f"[INFO] Saved difficulty scores → {scores_path}")

    # Build hard subset: top hard_ratio by loss, optionally excluding extreme top (noisy)
    sorted_by_loss = sorted(range(len(losses)), key=lambda i: losses[i], reverse=True)
    n_total = len(losses)
    n_hard = int(n_total * args.hard_ratio)
    n_skip_top = int(n_total * args.min_loss_pct)  # skip the absolute hardest (likely noisy/ambiguous)
    hard_indices = sorted_by_loss[n_skip_top: n_skip_top + n_hard]
    hard_indices = sorted(hard_indices)

    hard_ds = raw_ds.select(hard_indices)
    hard_dir = os.path.join(prepared_dir, "hard_subset")
    hard_ds.save_to_disk(hard_dir)

    avg_all = sum(losses) / len(losses)
    avg_hard = sum(losses[i] for i in hard_indices) / len(hard_indices)
    print(f"[INFO] All samples  — avg loss: {avg_all:.4f}")
    print(f"[INFO] Hard subset  — {len(hard_ds)} samples ({args.hard_ratio*100:.0f}%), avg loss: {avg_hard:.4f}")
    print(f"[INFO] Hard subset saved → {hard_dir}")
    print("[INFO] Set use_hard_mining: true in config to use this subset during training.")


if __name__ == "__main__":
    main()
