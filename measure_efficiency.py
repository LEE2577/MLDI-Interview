#!/usr/bin/env python
"""
Measure test-time latency and FLOPs for a single Qwen3-VL model.
Run once per model in separate processes to ensure a fair CUDA cold-start baseline.

Usage:
    # Step 1: measure each model independently (run in separate shell invocations)
    python measure_efficiency.py --model ./models/Qwen3-VL-2B-Instruct --save base.json
    python measure_efficiency.py --model ./outputs/.../merged           --save ft.json

    # Step 2: compare saved results
    python measure_efficiency.py --compare base.json ft.json
"""
import argparse
import json
import statistics
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Install qwen-vl-utils: pip install qwen-vl-utils")


WARMUP_RUNS = 5   # enough to compile CUDA kernels
MEASURE_RUNS = 20
MAX_NEW_TOKENS = 20
# Fixed canonical input: 336×336 grey image, within min/max_pixels used in eval
_CANONICAL_IMAGE = Image.new("RGB", (336, 336), color=(128, 128, 128))
_CANONICAL_QUESTION = "What text can you see in this image?"
_MAX_PIXELS = 200704
_MIN_PIXELS = 100352


def load_model(model_path, device):
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_path,
        max_pixels=_MAX_PIXELS,
        min_pixels=_MIN_PIXELS,
    )
    return model, processor


def build_inputs(processor, device):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": _CANONICAL_IMAGE,
                    "max_pixels": _MAX_PIXELS,
                    "min_pixels": _MIN_PIXELS,
                },
                {"type": "text", "text": _CANONICAL_QUESTION},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, _ = process_vision_info(
        messages, return_video_kwargs=True, image_patch_size=16
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        do_resize=False,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items()}


def measure_latency(model, inputs, n_warmup, n_runs):
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)

    # Warmup: ensures CUDA kernels are compiled before timing starts
    for _ in range(n_warmup):
        with torch.no_grad():
            model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return statistics.median(times), statistics.stdev(times)


def measure_flops(model, inputs):
    gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=True,
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            model.generate(**inputs, **gen_kwargs)
    return sum(e.flops for e in prof.key_averages())


def cmd_measure(args):
    """Measure a single model and optionally save results to JSON."""
    print(f"Loading: {args.model}")
    model, processor = load_model(args.model, args.device)
    inputs = build_inputs(processor, args.device)

    seq_len = inputs["input_ids"].shape[1]
    print(f"  Input sequence length (image + text tokens): {seq_len}")
    print(f"  Latency: {args.warmup} warmup + {args.runs} timed runs ...")
    lat, std = measure_latency(model, inputs, args.warmup, args.runs)
    print(f"  Latency: {lat * 1000:.1f} ms  (std {std * 1000:.1f} ms)")

    flops = None
    if not args.skip_flops:
        print("  FLOPs: profiling ...")
        flops = measure_flops(model, inputs)
        print(f"  FLOPs: {flops / 1e9:.1f} GFLOPs")

    result = {
        "model_path": args.model,
        "seq_len": seq_len,
        "latency_median_s": lat,
        "latency_std_s": std,
        "flops": flops,
        "warmup_runs": args.warmup,
        "measure_runs": args.runs,
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {args.save}")

    return result


def _ratio_label(ratio):
    status = "PASS" if ratio <= 1.1 else "FAIL"
    return f"{ratio:.4f}x  [{status}]"


def cmd_compare(args):
    """Load two saved JSON results and print the ratio report."""
    with open(args.compare[0]) as f:
        base = json.load(f)
    with open(args.compare[1]) as f:
        ft = json.load(f)

    col = 48

    def flops_str(f):
        return f"{f / 1e9:.1f}" if f is not None else "N/A"

    print("\n" + "=" * 76)
    print("EFFICIENCY REPORT  (each model measured in an isolated process)")
    print("=" * 76)
    print(f"{'Model':<{col}} {'Latency (ms)':>14}  {'FLOPs (GFLOPs)':>14}")
    print("-" * 76)
    print(f"{'[base] ' + base['model_path']:<{col}} {base['latency_median_s'] * 1000:>12.1f}ms  {flops_str(base['flops']):>14}")
    print(f"{'[ft]   ' + ft['model_path']:<{col}} {ft['latency_median_s'] * 1000:>12.1f}ms  {flops_str(ft['flops']):>14}")
    print("-" * 76)

    lat_ratio = ft["latency_median_s"] / base["latency_median_s"]
    print(f"\nLatency ratio:  {_ratio_label(lat_ratio)}")

    overall_pass = lat_ratio <= 1.1
    if base["flops"] is not None and ft["flops"] is not None:
        flops_ratio = ft["flops"] / base["flops"]
        print(f"FLOPs ratio:    {_ratio_label(flops_ratio)}")
        overall_pass = overall_pass and flops_ratio <= 1.1

    print(f"\nBudget check (≤1.1x): {'PASS ✓' if overall_pass else 'FAIL ✗'}")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", metavar="PATH", help="Model to measure")
    mode.add_argument("--compare", nargs=2, metavar=("BASE_JSON", "FT_JSON"),
                      help="Compare two saved result JSON files")

    parser.add_argument("--save", metavar="PATH", help="Save measurement result to JSON")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_flops", action="store_true",
                        help="Skip FLOPs measurement (saves ~30s)")
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=MEASURE_RUNS)
    args = parser.parse_args()

    if args.compare:
        cmd_compare(args)
    else:
        cmd_measure(args)


if __name__ == "__main__":
    main()
