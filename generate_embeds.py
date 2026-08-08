"""
flux2tiny — Teacher embedding generator for existing synthetic datasets.

Reads manifest.json from an existing dataset directory (containing latents and prompts),
computes teacher hidden states (Qwen3-4B layers 8, 18, 28 = 7680 dim), saves embed_000000.pt
files, and updates manifest.json with "embed_file" references.

Usage:
  python generate_embeds.py --dataset-dir synthetic_sd_69267 --batch-size 32
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from diffusers import Flux2KleinPipeline
from config import get_default_dtype


def generate_teacher_embeds(
    dataset_dir: str = "synthetic_sd_69267",
    flux_model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    max_seq_len: int = 128,
    teacher_extract_layers: tuple = (8, 18, 28),
    fp16: bool = False,
    batch_size: int = 32,
):
    dataset_path = Path(dataset_dir)
    manifest_path = dataset_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in '{dataset_dir}'")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if fp16 else get_default_dtype()

    print(f"=== Generating Teacher Embeddings for '{dataset_dir}' ===")
    print(f"  Teacher Model: {flux_model_id} | Total Items: {len(manifest):,} | Dtype: {dtype}")

    print(f"\nLoading FLUX.2 Qwen3 text encoder...")
    pipe = Flux2KleinPipeline.from_pretrained(
        flux_model_id, transformer=None, vae=None, dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    start_time = time.time()
    num_items = len(manifest)

    # Process in batches for maximum GPU throughput
    pbar = tqdm(range(0, num_items, batch_size), desc="Generating Embeddings")
    for i in pbar:
        batch_items = manifest[i : i + batch_size]
        prompts = [item["prompt"] for item in batch_items]

        # Tokenize and extract teacher hidden states
        inputs = pipe.tokenizer(
            prompts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=max_seq_len,
        ).to(device)

        with torch.no_grad():
            outputs = pipe.text_encoder(**inputs, output_hidden_states=True, return_dict=True)
            teacher_hidden = [outputs.hidden_states[layer + 1] for layer in teacher_extract_layers]
            batch_embeds = torch.cat(teacher_hidden, dim=-1)  # [B, max_seq_len, 7680]

        for item, embed in zip(batch_items, batch_embeds):
            embed_filename = f"embed_{item['id']:06d}.pt"
            torch.save(embed.cpu(), dataset_path / embed_filename)
            item["embed_file"] = embed_filename

        # Save manifest periodically
        if (i // batch_size + 1) % 50 == 0 or (i + batch_size) >= num_items:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n=== Done: {num_items:,} teacher embeddings saved in {elapsed/60:.1f} min ({elapsed/num_items:.3f} s/item) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate teacher embeddings for existing synthetic datasets")
    parser.add_argument("--dataset-dir", type=str, default="synthetic_sd_69267", help="Path to dataset folder with manifest.json")
    parser.add_argument("--flux-model", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fp16", action="store_true", help="Force float16 precision")
    args = parser.parse_args()

    generate_teacher_embeds(
        dataset_dir=args.dataset_dir,
        flux_model_id=args.flux_model,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        fp16=args.fp16,
    )
