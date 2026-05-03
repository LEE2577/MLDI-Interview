#!/usr/bin/env python
import argparse
import json
import os
import time

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForVision2Seq, AutoProcessor, Trainer, TrainerCallback, TrainingArguments, set_seed


DEFAULT_CONFIG = "configs/vlm_textvqa_lora.yaml"
SYSTEM_PROMPT = "You are a helpful assistant."


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(os.getenv("SEED", cfg.get("seed", 1)))
    cfg["seed"] = seed
    cfg["output_dir"] = os.getenv("OUTPUT_DIR", cfg["output_dir"]).format(seed=seed)
    cfg["prepared_data_dir"] = os.getenv("PREPARED_DATA_DIR", cfg["prepared_data_dir"]).format(seed=seed)
    return cfg


def load_prepared_dataset(cfg):
    if not os.path.isdir(cfg["prepared_data_dir"]):
        raise FileNotFoundError(
            f"Prepared dataset not found: {cfg['prepared_data_dir']}. "
            f"Run `python prepare_textvqa.py --config {DEFAULT_CONFIG}` first."
        )
    ds = load_from_disk(cfg["prepared_data_dir"])
    print(f"[INFO] Loaded {len(ds)} prepared TextVQA samples from {cfg['prepared_data_dir']}")
    return ds


class TimeLimitCallback(TrainerCallback):
    def __init__(self, max_seconds):
        self.max_seconds = max_seconds
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.max_seconds > 0 and time.time() - self.start_time > self.max_seconds:
            print(f"[TIMEOUT] Reached {self.max_seconds / 60:.1f} minute training budget")
            control.should_training_stop = True
        return control


class TextVQADataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, processor, cfg):
        self.ds = hf_ds
        self.processor = processor
        self.cfg = cfg

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"].convert("RGB")
        answer = item["target_answer"]
        user_text = item["user_text"]

        prompt_conv = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
        ]
        full_conv = prompt_conv + [{"role": "assistant", "content": answer}]

        prompt_text = self.processor.apply_chat_template(prompt_conv, tokenize=False, add_generation_prompt=True)
        full_text = self.processor.apply_chat_template(full_conv, tokenize=False, add_generation_prompt=False)

        common_kwargs = dict(
            images=[image],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=int(self.cfg.get("max_seq_length", 1024)),
        )
        prompt_batch = self.processor(text=prompt_text, **common_kwargs)
        full_batch = self.processor(text=full_text, **common_kwargs)

        input_ids = full_batch["input_ids"][0]
        attention_mask = full_batch["attention_mask"][0]
        labels = input_ids.clone()
        labels[: min(prompt_batch["input_ids"].shape[1], labels.shape[0])] = -100

        result = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if "pixel_values" in full_batch:
            result["pixel_values"] = full_batch["pixel_values"]
        if "image_grid_thw" in full_batch:
            result["image_grid_thw"] = full_batch["image_grid_thw"]
        return result


def collate_fn(examples, processor):
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    batch = {
        "input_ids": pad_sequence([ex["input_ids"] for ex in examples], batch_first=True, padding_value=pad_id),
        "attention_mask": pad_sequence([ex["attention_mask"] for ex in examples], batch_first=True, padding_value=0),
        "labels": pad_sequence([ex["labels"] for ex in examples], batch_first=True, padding_value=-100),
    }

    if "pixel_values" in examples[0]:
        batch["pixel_values"] = torch.cat([ex["pixel_values"] for ex in examples], dim=0)
    if "image_grid_thw" in examples[0]:
        batch["image_grid_thw"] = torch.cat([ex["image_grid_thw"] for ex in examples], dim=0)
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    processor = AutoProcessor.from_pretrained(
        cfg["model_path"],
        trust_remote_code=True,
        max_pixels=int(cfg["max_pixels"]),
        min_pixels=int(cfg["min_pixels"]),
    )
    model = AutoModelForVision2Seq.from_pretrained(
        cfg["model_path"],
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation=cfg.get("attn_implementation", "eager"),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if hasattr(model, "visual"):
        for param in model.visual.parameters():
            param.requires_grad = False

    lora_config = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    raw_ds = load_prepared_dataset(cfg)
    train_ds = TextVQADataset(raw_ds, processor, cfg)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        max_steps=int(cfg["max_steps"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        weight_decay=float(cfg["weight_decay"]),
        logging_steps=int(cfg["logging_steps"]),
        save_strategy="no",
        fp16=True,
        bf16=False,
        dataloader_num_workers=int(cfg["dataloader_num_workers"]),
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=lambda examples: collate_fn(examples, processor),
        callbacks=[TimeLimitCallback(int(cfg.get("max_train_seconds", 0)))],
    )
    trainer.train()

    final_dir = os.path.join(cfg["output_dir"], "final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
