#!/usr/bin/env python
import argparse
import json
import os
from collections import Counter
from glob import glob

import yaml
from datasets import Dataset

from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


DEFAULT_CONFIG = "configs/vlm_textvqa_lora.yaml"
EVAL_ANSWER_PROCESSOR = EvalAIAnswerProcessor()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(os.getenv("SEED", cfg.get("seed", 1)))
    cfg["seed"] = seed
    cfg["prepared_data_dir"] = os.getenv("PREPARED_DATA_DIR", cfg["prepared_data_dir"]).format(seed=seed)
    return cfg


def normalize_answer(answer):
    return EVAL_ANSWER_PROCESSOR(answer)


def textvqa_accuracy(candidate, references):
    scores = []
    for i in range(len(references)):
        other_answers = [references[j] for j in range(len(references)) if i != j]
        scores.append(min(1.0, other_answers.count(candidate) / 3.0))
    return sum(scores) / len(scores) if scores else 0.0


def choose_answer(answers):
    if not isinstance(answers, list):
        return normalize_answer(answers)
    normalized = [normalize_answer(ans) for ans in answers if str(ans).strip()]
    if not normalized:
        return ""
    counts = Counter(normalized)
    return max(counts, key=lambda ans: (textvqa_accuracy(ans, normalized), counts[ans], -normalized.index(ans)))


def build_question(item, use_ocr, max_ocr_tokens):
    question = item["question"].strip().capitalize()
    if use_ocr:
        seen = set()
        ocr_tokens = [str(tok) for tok in item.get("ocr_tokens", []) if str(tok).strip() and not (str(tok).lower() in seen or seen.add(str(tok).lower()))]
        if max_ocr_tokens > 0:
            ocr_tokens = ocr_tokens[:max_ocr_tokens]
        if ocr_tokens:
            ocr_text = " ".join(ocr_tokens)
            return f"The image contains the following text: {ocr_text}.\n{question}\nAnswer the question using a single word or phrase."
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
        answers = item.get("answers", [])
        if not isinstance(answers, list):
            answers = [answers]
        normalized = [normalize_answer(a) for a in answers if str(a).strip()]
        # deduplicate preserving order
        seen = set()
        deduped = [a for a in normalized if not (a in seen or seen.add(a))]
        item["target_answer"] = choose_answer(item["answers"])
        item["all_answers"] = deduped if deduped else [item["target_answer"]]
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
