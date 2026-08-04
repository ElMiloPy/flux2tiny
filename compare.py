#!/usr/bin/env python3
"""
flux2tiny — Side-by-side comparison: original FLUX.2 vs. flux2tiny.

Usage:
  python compare.py --prompt "A cat holding a sign" --config configs/minicpm5-1b.json
"""

import argparse
import time
from pathlib import Path

import torch
from config import add_config_argument, get_student_config, get_default_dtype


def measure_vram():
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0


def generate_original(prompt, h, w, steps, seed, dtype):
    """Generate with the original Flux2KleinPipeline (Qwen3-4B)."""
    from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2

    torch.cuda.reset_peak_memory_stats()
    vae = AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-small-decoder", torch_dtype=dtype)
    pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", vae=vae, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(seed)
    start = time.time()
    image = pipe(prompt=prompt, height=h, width=w, guidance_scale=1.0,
                 num_inference_steps=steps, generator=generator).images[0]
    elapsed = time.time() - start
    vram = torch.cuda.max_memory_allocated() / 1024**2

    del pipe
    torch.cuda.empty_cache()
    return image, elapsed, vram


def generate_tiny(prompt, h, w, steps, seed, config, dtype):
    """Generate with flux2tiny pipeline."""
    from pipeline import Flux2TinyPipeline

    torch.cuda.reset_peak_memory_stats()
    pipe = Flux2TinyPipeline(config=config, dtype=dtype, cpu_offload=True)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    start = time.time()
    image = pipe(prompt=prompt, height=h, width=w,
                 num_inference_steps=steps, guidance_scale=1.0, generator=generator)
    elapsed = time.time() - start
    vram = torch.cuda.max_memory_allocated() / 1024**2

    del pipe
    torch.cuda.empty_cache()
    return image, elapsed, vram


def main():
    parser = argparse.ArgumentParser(description="Compare original vs flux2tiny")
    add_config_argument(parser)
    parser.add_argument("--prompt", type=str, default="A cat holding a sign that says hello world")
    parser.add_argument("--size", type=str, default="512x512")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="comparisons")
    args = parser.parse_args()

    w, h = map(int, args.size.lower().split("x"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.fp16 else get_default_dtype()
    cfg = get_student_config(args.config)

    print(f"Prompt: {args.prompt}")
    print(f"Size: {w}x{h} | Steps: {args.steps} | Seed: {args.seed} | Config: {cfg.name}\n")

    # --- Original ---
    print("=" * 50)
    print("ORIGINAL (Qwen3-4B)")
    print("=" * 50)
    try:
        img_orig, t_orig, vram_orig = generate_original(args.prompt, h, w, args.steps, args.seed, dtype)
        img_orig.save(out / "original.png")
        print(f"  Time: {t_orig:.2f}s | Peak VRAM: {vram_orig:.0f}MB")
    except Exception as e:
        print(f"  Failed: {e}")
        img_orig, t_orig, vram_orig = None, 0, 0

    # --- Tiny ---
    print()
    print("=" * 50)
    print(f"FLUX2TINY ({cfg.name})")
    print("=" * 50)
    try:
        img_tiny, t_tiny, vram_tiny = generate_tiny(args.prompt, h, w, args.steps, args.seed, cfg, dtype)
        img_tiny.save(out / "flux2tiny.png")
        print(f"  Time: {t_tiny:.2f}s | Peak VRAM: {vram_tiny:.0f}MB")
    except Exception as e:
        print(f"  Failed: {e}")
        img_tiny, t_tiny, vram_tiny = None, 0, 0

    # --- Summary ---
    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    if vram_orig > 0 and vram_tiny > 0:
        print(f"  VRAM saved: {vram_orig - vram_tiny:.0f}MB ({(1 - vram_tiny/vram_orig)*100:.1f}%)")
    if t_orig > 0 and t_tiny > 0:
        print(f"  Time diff:  {t_tiny - t_orig:+.2f}s")

    if img_orig and img_tiny:
        from PIL import Image
        side = Image.new("RGB", (w * 2 + 20, h + 40), (30, 30, 30))
        side.paste(img_orig, (0, 40))
        side.paste(img_tiny, (w + 20, 40))
        side.save(out / "side_by_side.png")
        print(f"  Side-by-side: {out}/side_by_side.png")


if __name__ == "__main__":
    main()
