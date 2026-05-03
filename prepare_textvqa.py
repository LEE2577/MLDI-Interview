#!/usr/bin/env python
import argparse
import json
import os
import re
import string
from collections import Counter
from glob import glob

import yaml
from datasets import Dataset


DEFAULT_CONFIG = "configs/vlm_textvqa_lora.yaml"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(os.getenv("SEED", cfg.get("seed", 1)))
    cfg["seed"] = seed
    cfg["prepared_data_dir"] = os.getenv("PREPARED_DATA_DIR", cfg["prepared_data_dir"]).format(seed=seed)
    return cfg


def normalize_answer(answer):
    answer = str(answer).lower().strip()
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", answer)


def choose_answer(answers):
    if not isinstance(answers, list):
        return normalize_answer(answers)
    normalized = [normalize_answer(ans) for ans in answers if str(ans).strip()]
    if not normalized:
        return ""
    return Counter(normalized).most_common(1)[0][0]


def build_question(item, use_ocr, max_ocr_tokens):
    question = item["question"].strip().capitalize()
    if use_ocr:
        ocr_tokens = [str(tok) for tok in item.get("ocr_tokens", []) if str(tok).strip()]
        if max_ocr_tokens > 0:
            ocr_tokens = ocr_tokens[:max_ocr_tokens]
        if ocr_tokens:
            question += "\nReference OCR token: " + ", ".join(ocr_tokens)
    return question + "\nAnswer the question using a single word or phrase."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)

    files = sorted(glob(cfg["data_path"]))
    if not files:
        raise FileNotFoundError(f"No parquet files matched data_path={cfg['data_path']}")

    ds = Dataset.from_parquet(files)
    ds = ds.shuffle(seed=cfg["seed"])
    max_samples = int(cfg.get("max_train_samples", 0))
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    use_ocr = bool(cfg.get("use_ocr_tokens", True))
    max_ocr_tokens = int(cfg.get("max_ocr_tokens", 16))

    def add_training_fields(item):
        item["target_answer"] = choose_answer(item["answers"])
        item["user_text"] = build_question(item, use_ocr, max_ocr_tokens)
        return item

    ds = ds.map(add_training_fields, desc="Preparing TextVQA prompts")
    os.makedirs(os.path.dirname(cfg["prepared_data_dir"]), exist_ok=True)
    ds.save_to_disk(cfg["prepared_data_dir"])

    manifest = {
        "source_data_path": cfg["data_path"],
        "prepared_data_dir": cfg["prepared_data_dir"],
        "seed": cfg["seed"],
        "num_samples": len(ds),
        "use_ocr_tokens": use_ocr,
        "max_ocr_tokens": max_ocr_tokens,
    }
    with open(os.path.join(cfg["prepared_data_dir"], "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[INFO] Prepared {len(ds)} samples at {cfg['prepared_data_dir']}")


if __name__ == "__main__":
    main()
