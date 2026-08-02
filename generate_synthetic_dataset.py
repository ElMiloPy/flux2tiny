"""
flux2tiny — Teacher Synthetic Latent Generator

Generates high-quality synthetic latent-prompt pairs directly from the original
FLUX.2-klein-4B model (Qwen3-4B teacher) for student LoRA distillation.

Saves latent tensors directly to disk without slow PNG decoding/encoding cycles.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2


# Diverse prompt templates if no dataset prompt list is provided
FALLBACK_PROMPT_TEMPLATES = [
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


def generate_prompt_list(count: int = 15000, dataset_name: str = "Gustavosta/Stable-Diffusion-Prompts") -> list[str]:
    """Fetch real image generation prompts from Hugging Face datasets."""
    prompts = []
    import random
    random.seed(42)

    try:
        print(f"Fetching prompts from Hugging Face ({dataset_name})...")
        ds = load_dataset(dataset_name, split="train")
        for item in ds:
            # Check common key names across prompt datasets
            p = item.get("Prompt", item.get("text", item.get("prompt", "")))
            if isinstance(p, str) and len(p.strip()) > 5:
                prompts.append(p.strip())
            if len(prompts) >= count:
                break
        print(f"Successfully loaded {len(prompts)} prompts from {dataset_name}")
    except Exception as e:
        print(f"Could not load HF prompt dataset '{dataset_name}': {e}")

    # Fill remaining with structured random prompts if needed
    while len(prompts) < count:
        tmpl = random.choice(FALLBACK_PROMPT_TEMPLATES)
        p = tmpl.format(
            animal=random.choice(ANIMALS),
            setting=random.choice(SETTINGS),
            time=random.choice(TIMES),
            object=random.choice(OBJECTS),
            subject=random.choice(SUBJECTS),
        )
        prompts.append(p)

    return prompts[:count]


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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=== Generating Synthetic Teacher Latents ===")
    print(f"Target Samples: {num_samples}")
    print(f"Prompt Dataset: {prompt_dataset}")
    print(f"Resolution: {img_size}x{img_size}, Steps: {steps}")
    print(f"Saving directory: {output_path.resolve()}")

    prompts = generate_prompt_list(num_samples, dataset_name=prompt_dataset)

    # Load original FLUX.2 Klein pipeline (with native Qwen3 text encoder)
    print("\nLoading original FLUX.2 pipeline (Qwen3 teacher + FLUX.2 Transformer)...")
    vae = AutoencoderKLFlux2.from_pretrained(vae_model_id, torch_dtype=dtype)
    pipe = Flux2KleinPipeline.from_pretrained(flux_model_id, vae=vae, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)

    manifest = []
    start_time = time.time()

    for idx, prompt in enumerate(tqdm(prompts, desc="Generating Teacher Latents")):
        seed = seed_start + idx
        generator = torch.Generator(device=device).manual_seed(seed)

        # Generate latents using FLUX.2 pipeline (intercepting before VAE decode)
        with torch.no_grad():
            latents = pipe(
                prompt=prompt,
                height=img_size,
                width=img_size,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="latent",
            ).images[0]

        latent_filename = f"latent_{idx:06d}.pt"
        torch.save(latents.cpu(), output_path / latent_filename)

        manifest.append({
            "id": idx,
            "prompt": prompt,
            "latent_file": latent_filename,
            "seed": seed,
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == num_samples:
            with open(output_path / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

    elapsed = time.time() - start_time
    avg_speed = elapsed / num_samples
    print(f"\n=== Completed generating {num_samples} synthetic latents in {elapsed/60:.2f} mins ===")
    print(f"Average speed: {avg_speed:.2f} sec/sample")
    print(f"Dataset manifest saved to: {output_path / 'manifest.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic teacher latents with FLUX.2")
    parser.add_argument("--output-dir", type=str, default="synthetic_dataset")
    parser.add_argument("--prompt-dataset", type=str, default="Gustavosta/Stable-Diffusion-Prompts",
                        help="HuggingFace dataset for prompts (e.g., Gustavosta/Stable-Diffusion-Prompts or succinctly/midjourney-prompts)")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=512)
    args = parser.parse_args()

    generate_synthetic_latents(
        output_dir=args.output_dir,
        prompt_dataset=args.prompt_dataset,
        num_samples=args.num_samples,
        steps=args.steps,
        img_size=args.img_size,
    )
