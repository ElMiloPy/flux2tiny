#!/usr/bin/env python3
"""
flux2tiny — Compare outputs between original and adapted pipelines.

Generates the same prompt with the same seed using:
  1. Original Flux2KleinPipeline (Qwen3-4B text encoder)
  2. flux2tiny pipeline (MiniCPM5-1B + adapter)

Outputs a side-by-side comparison and VRAM usage stats.
"""

import argparse
import time
from pathlib import Path

import torch


def measure_vram():
    """Return current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def generate_original(prompt, height, width, steps, seed):
    """Generate with the original Flux2KleinPipeline."""
    from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2

    vram_before = measure_vram()

    vae = AutoencoderKLFlux2.from_pretrained(
        "black-forest-labs/FLUX.2-small-decoder", torch_dtype=torch.bfloat16
    )
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    vram_loaded = measure_vram()
    generator = torch.Generator(device="cuda").manual_seed(seed)

    start = time.time()
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=1.0,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]
    elapsed = time.time() - start

    vram_peak = torch.cuda.max_memory_allocated() / 1024**2

    # Clean up
    del pipe
    torch.cuda.empty_cache()

    return image, elapsed, vram_peak


def generate_tiny(prompt, height, width, steps, seed, adapter_path):
    """Generate with flux2tiny pipeline."""
    from pipeline import Flux2TinyPipeline

    torch.cuda.reset_peak_memory_stats()
    vram_before = measure_vram()

    pipe = Flux2TinyPipeline(
        adapter_path=adapter_path,
        cpu_offload=True,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)

    start = time.time()
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=1.0,
        generator=generator,
    )
    elapsed = time.time() - start

    vram_peak = torch.cuda.max_memory_allocated() / 1024**2

    del pipe
    torch.cuda.empty_cache()

    return image, elapsed, vram_peak


def main():
    parser = argparse.ArgumentParser(description="Compare original vs flux2tiny")
    parser.add_argument("--prompt", type=str, default="A cat holding a sign that says hello world")
    parser.add_argument("--size", type=str, default="512x512")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapter", type=str, default="adapter_checkpoints/adapter_best.safetensors")
    parser.add_argument("--output-dir", type=str, default="comparisons")
    args = parser.parse_args()

    w, h = map(int, args.size.lower().split("x"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Prompt: {args.prompt}")
    print(f"Size: {w}x{h}, Steps: {args.steps}, Seed: {args.seed}")
    print()

    # --- Original ---
    print("=" * 50)
    print("ORIGINAL (Qwen3-4B)")
    print("=" * 50)
    try:
        torch.cuda.reset_peak_memory_stats()
        img_orig, time_orig, vram_orig = generate_original(
            args.prompt, h, w, args.steps, args.seed
        )
        img_orig.save(output_dir / "original.png")
        print(f"  Time: {time_orig:.2f}s | Peak VRAM: {vram_orig:.0f}MB")
    except Exception as e:
        print(f"  Failed: {e}")
        img_orig, time_orig, vram_orig = None, 0, 0

    # --- Tiny ---
    print()
    print("=" * 50)
    print("FLUX2TINY (MiniCPM5-1B + adapter)")
    print("=" * 50)
    try:
        torch.cuda.reset_peak_memory_stats()
        img_tiny, time_tiny, vram_tiny = generate_tiny(
            args.prompt, h, w, args.steps, args.seed, args.adapter
        )
        img_tiny.save(output_dir / "flux2tiny.png")
        print(f"  Time: {time_tiny:.2f}s | Peak VRAM: {vram_tiny:.0f}MB")
    except Exception as e:
        print(f"  Failed: {e}")
        img_tiny, time_tiny, vram_tiny = None, 0, 0

    # --- Summary ---
    print()
    print("=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    if vram_orig > 0 and vram_tiny > 0:
        print(f"  VRAM saved:  {vram_orig - vram_tiny:.0f}MB ({(1 - vram_tiny/vram_orig)*100:.1f}%)")
    if time_orig > 0 and time_tiny > 0:
        print(f"  Time diff:   {time_tiny - time_orig:+.2f}s")
    print(f"  Outputs in:  {output_dir}/")

    # Side-by-side
    if img_orig and img_tiny:
        from PIL import Image
        comparison = Image.new("RGB", (w * 2 + 20, h + 40), (30, 30, 30))
        comparison.paste(img_orig, (0, 40))
        comparison.paste(img_tiny, (w + 20, 40))
        comparison.save(output_dir / "side_by_side.png")
        print(f"  Side-by-side: {output_dir}/side_by_side.png")


if __name__ == "__main__":
    main()
