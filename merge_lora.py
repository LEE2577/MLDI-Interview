#!/usr/bin/env python
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="/storage/data/shiyd2023/LLM_model/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter", default="./outputs/textvqa_qwen3vl_lora_seed1/final")
    parser.add_argument("--output", default="./outputs/textvqa_qwen3vl_lora_seed1/merged")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    processor.save_pretrained(args.output)
    print(f"[INFO] Merged model saved to {args.output}")


if __name__ == "__main__":
    main()
