#!/usr/bin/env python3
"""
flux2tiny — Image generation CLI.

Usage:
  python generate.py "A cat holding a sign that says hello world"
  python generate.py "A sunset" --config configs/qwen3.5-0.8b.json --size 512x512 --seed 42
"""

import argparse
import time
from pathlib import Path

import torch
from pipeline import Flux2TinyPipeline
from config import add_config_argument, get_student_config


def main():
    parser = argparse.ArgumentParser(description="Generate images with flux2tiny")
    parser.add_argument("prompt", type=str, help="Text prompt")
    add_config_argument(parser)
    parser.add_argument("--adapter", type=str, default=None, help="Adapter weights path")
    parser.add_argument("--lora", type=str, default=None, help="LoRA weights path")
    parser.add_argument("--adapter-type", type=str, default="per_layer", choices=["per_layer", "concat"])
    parser.add_argument("--flux-model", type=str, default=None)
    parser.add_argument("--vae-model", type=str, default=None)
    parser.add_argument("--student-model", type=str, default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--size", type=str, default="1024x1024", help="WxH")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-cpu-offload", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=128)
    args = parser.parse_args()

    cfg = get_student_config(args.config)
    w, h = map(int, args.size.lower().split("x"))

    # Output path
    if args.output:
        out = Path(args.output)
    else:
        slug = args.prompt[:40].replace(" ", "_").replace("/", "-")
        seed_str = f"_s{args.seed}" if args.seed is not None else ""
        out = Path(f"output/{cfg.name}_{slug}{seed_str}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Seed
    generator = None
    if args.seed is not None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=dev).manual_seed(args.seed)

    print(f"Config: {cfg.name} | Size: {w}x{h} | Steps: {args.steps} | Seed: {args.seed}")
    print(f"Prompt: {args.prompt}\n")

    dtype = torch.float16 if args.fp16 else None

    pipe = Flux2TinyPipeline(
        config=cfg,
        flux_model_id=args.flux_model,
        vae_model_id=args.vae_model,
        student_model_id=args.student_model,
        adapter_path=args.adapter,
        lora_path=args.lora,
        adapter_type=args.adapter_type,
        dtype=dtype,
        cpu_offload=not args.no_cpu_offload,
    )

    start = time.time()
    image = pipe(
        prompt=args.prompt, height=h, width=w,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
        max_seq_len=args.max_seq_len,
    )
    print(f"\nGenerated in {time.time() - start:.2f}s → {out}")
    image.save(str(out))


if __name__ == "__main__":
    main()
