#!/usr/bin/env python3
"""
flux2tiny — Image generation script.

Usage:
  python generate.py "A cat holding a sign that says hello world"
  python generate.py "A sunset over the ocean" --config configs/qwen3.5-0.8b.json --steps 4 --seed 42 --size 512x512
"""

import argparse
import time
from pathlib import Path

import torch
from pipeline import Flux2TinyPipeline
from config import add_config_argument, get_student_config


def main():
    parser = argparse.ArgumentParser(description="Generate images with flux2tiny")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    add_config_argument(parser)
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to trained adapter weights (defaults to config preset path)")
    parser.add_argument("--lora", type=str, default=None,
                        help="Path to trained transformer LoRA weights (defaults to config preset path)")
    parser.add_argument("--adapter-type", type=str, default="per_layer", choices=["per_layer", "concat"])
    parser.add_argument("--flux-model", type=str, default=None, help="Override FLUX.2 transformer model ID")
    parser.add_argument("--vae-model", type=str, default=None, help="Override VAE decoder model ID")
    parser.add_argument("--student-model", type=str, default=None, help="Override student text encoder model ID")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--size", type=str, default="1024x1024", help="Image size WxH")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output filename")
    parser.add_argument("--no-cpu-offload", action="store_true", help="Keep all models on GPU")
    parser.add_argument("--max-seq-len", type=int, default=128, help="Max token sequence length")
    args = parser.parse_args()

    student_cfg = get_student_config(args.config)

    # Parse size
    w, h = map(int, args.size.lower().split("x"))

    # Output filename
    if args.output:
        output_path = Path(args.output)
    else:
        slug = args.prompt[:40].replace(" ", "_").replace("/", "-")
        seed_str = f"_s{args.seed}" if args.seed is not None else ""
        output_path = Path(f"output/{student_cfg.name}_{slug}{seed_str}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generator
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.seed)

    # Load pipeline
    print(f"Preset: {student_cfg.name} ({student_cfg.description})")
    print(f"Prompt: {args.prompt}")
    print(f"Size: {w}x{h}, Steps: {args.steps}, Seed: {args.seed}")
    print()

    dtype = torch.float16 if args.fp16 else None

    pipe = Flux2TinyPipeline(
        config=student_cfg,
        flux_model_id=args.flux_model,
        vae_model_id=args.vae_model,
        student_model_id=args.student_model,
        adapter_path=args.adapter,
        lora_path=args.lora,
        adapter_type=args.adapter_type,
        dtype=dtype,
        cpu_offload=not args.no_cpu_offload,
        multi_gpu=args.multi_gpu,
    )

    # Generate
    print(f"\nGenerating...")
    start = time.time()

    image = pipe(
        prompt=args.prompt,
        height=h,
        width=w,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
        max_seq_len=args.max_seq_len,
    )

    elapsed = time.time() - start
    print(f"Generated in {elapsed:.2f}s")

    # Save
    image.save(str(output_path))
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
