#!/usr/bin/env python3
"""
flux2tiny — Upload trained weights to Hugging Face Hub.

Uploads only the trained delta weights:
  1. Projection adapter (adapter.safetensors)
  2. Transformer LoRA (adapter_model.safetensors + config)

Users download base models (FLUX.2-klein-4B, student model, VAE) separately at runtime.
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from config import add_config_argument, get_student_config, StudentConfig


def generate_model_card(student_cfg: StudentConfig) -> str:
    return f"""---
license: mit
base_model:
  - {student_cfg.teacher_model_id}
  - {student_cfg.student_model_id}
  - {student_cfg.vae_model_id}
tags:
  - flux
  - flux2
  - distillation
  - lora
  - text-to-image
  - diffusers
library_name: diffusers
pipeline_tag: text-to-image
---

# flux2tiny — Distilled FLUX.2-klein-4B with {student_cfg.student_model_id} Text Encoder

This repository contains the **trained adapter and LoRA weights** for flux2tiny ({student_cfg.name}),
a distilled version of [{student_cfg.teacher_model_id}](https://huggingface.co/{student_cfg.teacher_model_id})
that replaces the 4B-parameter Qwen3-4B text encoder with
[{student_cfg.student_model_id}](https://huggingface.co/{student_cfg.student_model_id}) ({student_cfg.description}).

## What's in this repo

| File | Size | Description |
|:-----|:-----|:------------|
| `adapter.safetensors` | ~10-25 MB | Projection adapter (3× Linear {student_cfg.hidden_size}→2560, concatenated to 7680) |
| `transformer_lora/adapter_model.safetensors` | ~7.5 MB | PEFT LoRA weights (rank 16) for Flux2Transformer2DModel |
| `transformer_lora/adapter_config.json` | ~1 KB | PEFT LoRA configuration |

## Required base models (downloaded automatically)

- [{student_cfg.teacher_model_id}](https://huggingface.co/{student_cfg.teacher_model_id}) — Transformer backbone
- [{student_cfg.student_model_id}](https://huggingface.co/{student_cfg.student_model_id}) — Student text encoder
- [{student_cfg.vae_model_id}](https://huggingface.co/{student_cfg.vae_model_id}) — VAE decoder

## Usage

```python
# Clone the code repo
# git clone https://github.com/ElMiloPy/flux2tiny.git

from pipeline import Flux2TinyPipeline

pipe = Flux2TinyPipeline(
    config="{student_cfg.name}",
    adapter_path="path/to/adapter.safetensors",
    lora_path="path/to/transformer_lora",
)

image = pipe("A cat sitting on a windowsill at sunset", height=512, width=512)
image.save("output.png")
```

Or via CLI:
```bash
python generate.py "A cat sitting on a windowsill at sunset" \\
    --config {student_cfg.name} \\
    --adapter path/to/adapter.safetensors \\
    --lora path/to/transformer_lora \\
    --size 512x512
```

## Training details

Trained via a 3-stage knowledge distillation pipeline:

1. **Adapter pre-training** — MSE alignment between {student_cfg.student_model_id} and Qwen3-4B hidden states
2. **Teacher latent generation** — Synthetic latent-prompt pairs from original FLUX.2 pipeline
3. **Flow Matching LoRA distillation** — Joint training of adapter + transformer LoRA on teacher latents

See [github.com/ElMiloPy/flux2tiny](https://github.com/ElMiloPy/flux2tiny) for full details.

## License

- **These weights**: MIT
- **FLUX.2-klein-4B**: Apache 2.0
- **{student_cfg.student_model_id}**: Apache 2.0
"""


def upload_to_hub(
    checkpoint_dir: str,
    repo_id: str,
    student_cfg: StudentConfig,
    private: bool = False,
):
    """Upload trained weights to Hugging Face Hub."""
    ckpt_path = Path(checkpoint_dir)

    # Validate files exist
    adapter_file = ckpt_path / "adapter.safetensors"
    lora_dir = ckpt_path / "transformer_lora"

    if not adapter_file.exists():
        raise FileNotFoundError(f"Adapter weights not found: {adapter_file}")
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora_dir}")

    print(f"=== Uploading flux2tiny ({student_cfg.name}) weights to {repo_id} ===")
    print(f"  Adapter: {adapter_file} ({adapter_file.stat().st_size / 1e6:.1f} MB)")

    lora_model = lora_dir / "adapter_model.safetensors"
    if lora_model.exists():
        print(f"  LoRA:    {lora_model} ({lora_model.stat().st_size / 1e6:.1f} MB)")

    api = HfApi()

    # Create repo (model type)
    repo_url = create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"  Repo:    {repo_url}")

    # Upload adapter
    print("\nUploading adapter.safetensors...")
    api.upload_file(
        path_or_fileobj=str(adapter_file),
        path_in_repo="adapter.safetensors",
        repo_id=repo_id,
    )

    # Upload LoRA directory
    print("Uploading transformer_lora/...")
    api.upload_folder(
        folder_path=str(lora_dir),
        path_in_repo="transformer_lora",
        repo_id=repo_id,
        ignore_patterns=["README.md"],
    )

    # Upload model card
    print("Uploading README.md (model card)...")
    readme_path = ckpt_path / "_README_HF.md"
    readme_path.write_text(generate_model_card(student_cfg))
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    readme_path.unlink()  # Clean up temp file

    print(f"\n=== Done! Weights uploaded to: https://huggingface.co/{repo_id} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload flux2tiny weights to Hugging Face Hub")
    add_config_argument(parser)
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory containing adapter.safetensors and transformer_lora/")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="Hugging Face repo ID (e.g., 'Emilio407/flux2tiny-weights')")
    parser.add_argument("--private", action="store_true",
                        help="Make the repo private")
    args = parser.parse_args()

    student_cfg = get_student_config(args.config)
    checkpoint_dir = args.checkpoint_dir or student_cfg.get_lora_path("final")

    upload_to_hub(
        checkpoint_dir=checkpoint_dir,
        repo_id=args.repo_id,
        student_cfg=student_cfg,
        private=args.private,
    )
