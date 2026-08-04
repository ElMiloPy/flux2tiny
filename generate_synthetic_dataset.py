"""
flux2tiny — Teacher synthetic latent generator (Stage 2).

Generates latent-prompt pairs from the original FLUX.2-klein-4B pipeline
(with Qwen3-4B teacher) for downstream LoRA distillation.

Usage:
  python generate_synthetic_dataset.py --num-samples 15000 --output-dir synthetic_sd_15k
  python generate_synthetic_dataset.py --flux-model black-forest-labs/FLUX.2-klein-9B --num-samples 5000
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2

from config import get_default_dtype


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------
FALLBACK_TEMPLATES = [
    "A photo of a cute {animal} sitting in a {setting}, high detail, 4k",
    "A vibrant watercolor painting of a {setting} during {time}",
    "An oil painting of a {subject} with cinematic lighting, masterwork",
    "A sleek futuristic {object} in a glowing neon city at night",
    "A macro photograph of a {subject} with soft bokeh background",
]
ANIMALS = ["cat", "dog", "fox", "owl", "panda", "tiger", "rabbit", "lion", "koala", "penguin"]
SETTINGS = ["cozy living room", "lush green garden", "sunlit meadow", "snowy mountain", "misty forest", "sandy beach"]
TIMES = ["sunset", "sunrise", "starry night", "golden hour", "foggy morning"]
OBJECTS = ["sports car", "robot", "spaceship", "clockwork watch", "cyberpunk motorcycle"]
SUBJECTS = ["blooming flower", "dewdrop on a leaf", "butterfly", "crystal prism", "cup of steaming coffee"]


def fetch_prompts(count: int, dataset_name: str) -> list[str]:
    """Fetch prompts from HF dataset, filling remainder with random templates."""
    prompts = []
    random.seed(42)

    try:
        print(f"Fetching prompts from {dataset_name}...")
        ds = load_dataset(dataset_name, split="train")
        for item in ds:
            p = item.get("Prompt", item.get("text", item.get("prompt", "")))
            if isinstance(p, str) and len(p.strip()) > 5:
                prompts.append(p.strip())
            if len(prompts) >= count:
                break
        print(f"Loaded {len(prompts)} prompts")
    except Exception as e:
        print(f"Could not load '{dataset_name}': {e}")

    while len(prompts) < count:
        tmpl = random.choice(FALLBACK_TEMPLATES)
        prompts.append(tmpl.format(
            animal=random.choice(ANIMALS), setting=random.choice(SETTINGS),
            time=random.choice(TIMES), object=random.choice(OBJECTS),
            subject=random.choice(SUBJECTS),
        ))

    return prompts[:count]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_synthetic_latents(
    output_dir: str = "synthetic_dataset",
    prompt_dataset: str = "Gustavosta/Stable-Diffusion-Prompts",
    num_samples: int = 1000,
    steps: int = 4,
    guidance_scale: float = 1.0,
    img_size: int = 512,
    seed_start: int = 42,
    flux_model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    vae_model_id: str = "black-forest-labs/FLUX.2-small-decoder",
    fp16: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if fp16 else get_default_dtype()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=== Generating Synthetic Teacher Latents ===")
    print(f"  Model: {flux_model_id} | Dtype: {dtype}")
    print(f"  Samples: {num_samples} | Resolution: {img_size}x{img_size} | Steps: {steps}")

    prompts = fetch_prompts(num_samples, prompt_dataset)

    print(f"\nLoading FLUX.2 pipeline...")
    vae = AutoencoderKLFlux2.from_pretrained(vae_model_id, torch_dtype=dtype)
    pipe = Flux2KleinPipeline.from_pretrained(flux_model_id, vae=vae, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    manifest = []
    start_time = time.time()

    for idx, prompt in enumerate(tqdm(prompts, desc="Generating")):
        generator = torch.Generator(device=device).manual_seed(seed_start + idx)

        with torch.no_grad():
            latents = pipe(
                prompt=prompt, height=img_size, width=img_size,
                num_inference_steps=steps, guidance_scale=guidance_scale,
                generator=generator, output_type="latent",
            ).images[0]

        filename = f"latent_{idx:06d}.pt"
        torch.save(latents.cpu(), output_path / filename)
        manifest.append({"id": idx, "prompt": prompt, "latent_file": filename, "seed": seed_start + idx})

        if (idx + 1) % 100 == 0 or (idx + 1) == num_samples:
            with open(output_path / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n=== Done: {num_samples} latents in {elapsed/60:.1f} min ({elapsed/num_samples:.2f} s/sample) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic teacher latents (Stage 2)")
    parser.add_argument("--output-dir", type=str, default="synthetic_dataset")
    parser.add_argument("--prompt-dataset", type=str, default="Gustavosta/Stable-Diffusion-Prompts")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--flux-model", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--vae-model", type=str, default="black-forest-labs/FLUX.2-small-decoder")
    parser.add_argument("--fp16", action="store_true", help="Force float16 (for Pascal GPUs)")
    args = parser.parse_args()

    generate_synthetic_latents(
        output_dir=args.output_dir,
        prompt_dataset=args.prompt_dataset,
        num_samples=args.num_samples,
        steps=args.steps,
        img_size=args.img_size,
        flux_model_id=args.flux_model,
        vae_model_id=args.vae_model,
        fp16=args.fp16,
    )
