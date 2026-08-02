"""
flux2tiny — LoRA Training Script for FLUX.2 Transformer + Adapter

Trains a LoRA adapter on Flux2Transformer2DModel combined with the MiniCPM5-1B
projection adapter using Flow Matching Loss directly on image-text pairs.

This teaches the FLUX.2 transformer backbone how to interpret MiniCPM5-1B's
projected embeddings, producing coherent, high-quality images.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from adapter import PerLayerProjection, ConcatProjection, save_adapter, load_adapter
from config import get_student_config, add_config_argument, StudentConfig


# ---------------------------------------------------------------------------
# Synthetic / Downloaded Image-Caption Dataset
# ---------------------------------------------------------------------------
class ImageCaptionDataset(Dataset):
    """
    Dataset of image-caption pairs for Flow Matching training.
    If image_dir exists, loads local images + captions.
    Otherwise generates synthetic training patterns with descriptive prompts.
    """

    DEFAULT_PROMPTS = [
        "a vibrant red circle on a dark background",
        "a bright blue square centered on a white canvas",
        "a yellow star shining in a night sky",
        "a green leaf with detailed veins",
        "a colorful rainbow gradient across the screen",
        "a orange sunset over a black ocean horizon",
        "a purple sphere floating in space",
        "a geometric pattern of cyan and magenta triangles",
        "a golden retriever puppy sitting happily",
        "a cute fluffy cat looking at the camera",
    ]

    def __init__(
        self,
        image_dir: str | None = None,
        hf_dataset: str | None = None,
        synthetic_dir: str | None = None,
        img_size: int = 512,
        num_samples: int = 200,
    ):
        self.img_size = img_size
        self.samples = []
        self.hf_data = None
        self.synthetic_dir = None
        self.synthetic_manifest = []

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1] range for VAE
        ])

        if synthetic_dir and Path(synthetic_dir).exists():
            syn_path = Path(synthetic_dir)
            manifest_file = syn_path / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file, "r") as f:
                    self.synthetic_manifest = json.load(f)
                self.synthetic_dir = syn_path
                print(f"Loaded {len(self.synthetic_manifest)} synthetic latents from {synthetic_dir}")

        elif hf_dataset:
            from datasets import load_dataset
            print(f"Loading Hugging Face dataset: {hf_dataset}...")
            self.hf_data = load_dataset(hf_dataset, split="train")
            print(f"Loaded {len(self.hf_data)} samples from {hf_dataset}")

        elif image_dir and Path(image_dir).exists():
            image_path = Path(image_dir)
            valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
            for p in image_path.iterdir():
                if p.suffix.lower() in valid_exts:
                    txt_path = p.with_suffix(".txt")
                    caption = txt_path.read_text().strip() if txt_path.exists() else p.stem.replace("_", " ")
                    self.samples.append((str(p), caption))
            print(f"Loaded {len(self.samples)} image-caption pairs from {image_dir}")

        if not self.synthetic_dir and not self.hf_data and not self.samples:
            print(f"Generating {num_samples} synthetic training samples...")
            self.synthetic_data = []
            for i in range(num_samples):
                prompt = self.DEFAULT_PROMPTS[i % len(self.DEFAULT_PROMPTS)]
                # Create distinct synthetic color image
                img = self._create_synthetic_image(i, img_size)
                self.synthetic_data.append((img, prompt))

    def _create_synthetic_image(self, index: int, size: int) -> Image.Image:
        import numpy as np
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        color_idx = index % 5
        if color_idx == 0:
            arr[:, :, 0] = 220  # Red gradient
            arr[:, :, 1] = np.linspace(0, 200, size, dtype=np.uint8)
        elif color_idx == 1:
            arr[:, :, 2] = 240  # Blue gradient
            arr[:, :, 0] = np.linspace(0, 150, size, dtype=np.uint8)[:, None]
        elif color_idx == 2:
            arr[:, :, 0] = 240  # Yellow/Green
            arr[:, :, 1] = 220
        elif color_idx == 3:
            arr[:, :, 1] = 200  # Green
            arr[:, :, 2] = 180
        else:
            arr[:, :, 0] = 200  # Purple/Magenta
            arr[:, :, 2] = 220
        return Image.fromarray(arr)

    def __len__(self):
        if self.synthetic_dir:
            return len(self.synthetic_manifest)
        if self.hf_data is not None:
            return len(self.hf_data)
        if self.samples:
            return len(self.samples)
        return len(self.synthetic_data)

    def __getitem__(self, idx):
        if self.synthetic_dir:
            item = self.synthetic_manifest[idx]
            latent = torch.load(self.synthetic_dir / item["latent_file"], map_location="cpu")
            return {"latent": latent, "caption": item["prompt"]}
        elif self.hf_data is not None:
            item = self.hf_data[idx]
            img = item["image"].convert("RGB")
            caption = item.get("text", item.get("caption", "a photo"))
        elif self.samples:
            img_path, caption = self.samples[idx]
            img = Image.open(img_path).convert("RGB")
        else:
            img, caption = self.synthetic_data[idx]

        pixel_values = self.transform(img)
        return {"pixel_values": pixel_values, "caption": caption}


# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------
def train_lora(
    config: str | StudentConfig = "minicpm-1b",
    flux_model_id: str | None = None,
    vae_model_id: str | None = None,
    student_model_id: str | None = None,
    adapter_path: str | None = None,
    output_dir: str | None = None,
    image_dir: str | None = None,
    hf_dataset: str | None = None,
    synthetic_dir: str | None = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 1,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    img_size: int = 512,
):
    student_cfg = get_student_config(config) if isinstance(config, str) else config
    flux_model_id = flux_model_id or student_cfg.teacher_model_id
    vae_model_id = vae_model_id or student_cfg.vae_model_id
    student_model_id = student_model_id or student_cfg.student_model_id
    student_extract_layers = student_cfg.extract_layers

    if adapter_path is None:
        adapter_path = student_cfg.get_adapter_path("adapter_best.safetensors")
        if not Path(adapter_path).exists():
            adapter_path = student_cfg.get_adapter_path("adapter_final.safetensors")

    output_dir = output_dir or student_cfg.default_lora_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"=== Training FLUX.2 LoRA + Adapter (Preset: {student_cfg.name}) ===")
    print(f"Device: {device}")
    print(f"Image Size: {img_size}x{img_size}, LoRA Rank: {lora_rank}")

    # 1. Load Student Text Encoder — Frozen
    print(f"\nLoading student text encoder: {student_model_id}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(student_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_encoder = AutoModelForCausalLM.from_pretrained(
        student_model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    # 2. Load Adapter — Trainable
    print(f"\nLoading adapter from {adapter_path}...")
    adapter = load_adapter(
        adapter_path,
        adapter_type="per_layer",
        source_dim=student_cfg.hidden_size,
        target_dim=student_cfg.teacher_hidden_size,
        num_layers=student_cfg.num_layers,
        device=str(device),
        dtype=dtype,
    )
    adapter.train()
    for p in adapter.parameters():
        p.requires_grad = True

    # 3. Load VAE — Frozen
    print(f"\nLoading VAE: {vae_model_id}...")
    from diffusers import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2.from_pretrained(vae_model_id, torch_dtype=dtype).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # 4. Load Transformer & Apply PEFT LoRA
    print(f"\nLoading Transformer & applying PEFT LoRA: {flux_model_id}...")
    from diffusers.models import Flux2Transformer2DModel
    from peft import LoraConfig, get_peft_model

    transformer = Flux2Transformer2DModel.from_pretrained(
        flux_model_id, subfolder="transformer", torch_dtype=dtype
    ).to(device)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # 5. Dataset and Dataloader
    dataset = ImageCaptionDataset(image_dir=image_dir, hf_dataset=hf_dataset, synthetic_dir=synthetic_dir, img_size=img_size, num_samples=200)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 6. Optimizer (Adapter + LoRA parameters)
    trainable_params = list(adapter.parameters()) + list(transformer.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    total_steps = len(dataloader) * num_epochs
    print(f"\nTotal training steps: {total_steps}")
    print("=== Starting Flow Matching Training ===")

    global_step = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            captions = batch["caption"]

            # Encode Text Prompt via MiniCPM5-1B + Adapter
            inputs = tokenizer(
                captions, return_tensors="pt", padding="max_length",
                truncation=True, max_length=128
            ).to(device)

            with torch.no_grad():
                outputs = text_encoder(**inputs, output_hidden_states=True, return_dict=True)
                hidden_states_list = [outputs.hidden_states[idx + 1] for idx in student_extract_layers]

            prompt_embeds = adapter(hidden_states_list)  # [B, 128, 7680]

            # Latents: from pre-encoded synthetic batch or VAE encoding
            if "latent" in batch:
                latents_0 = batch["latent"].to(device=device, dtype=dtype)
                if latents_0.ndim == 3:
                    latents_0 = latents_0.unsqueeze(0)
            else:
                pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
                with torch.no_grad():
                    latents_0 = vae.encode(pixel_values).latent_dist.sample()

            # Helper functions for FLUX.2 packing
            B, C, H, W = latents_0.shape
            def pack_latents(l_tensor):
                x = l_tensor.view(B, C, H // 2, 2, W // 2, 2)
                return x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, (H // 2) * (W // 2)).permute(0, 2, 1)

            latents_0_packed = pack_latents(latents_0)  # [B, 1024, 128]

            # Sample random timesteps t in (0, 1) and random noise
            batch_size_cur = B
            t = torch.rand((batch_size_cur,), device=device, dtype=dtype)
            noise_0 = torch.randn_like(latents_0)
            noise_packed = pack_latents(noise_0)

            # Flow Matching interpolation on packed latents: z_t = (1 - t) * z_0 + t * noise
            t_expand = t.view(batch_size_cur, 1, 1)
            latents_t_packed = (1.0 - t_expand) * latents_0_packed + t_expand * noise_packed

            # Target velocity for Flow Matching: v = noise - latents_0
            target_velocity = noise_packed - latents_0_packed

            # Position IDs (4D: T, H, W, L)
            h_ids, w_ids = H // 2, W // 2
            grid = torch.cartesian_prod(
                torch.arange(1, device=device),
                torch.arange(h_ids, device=device),
                torch.arange(w_ids, device=device),
                torch.arange(1, device=device)
            ).to(dtype)
            img_ids = grid.unsqueeze(0).expand(batch_size_cur, -1, -1)  # [B, 1024, 4]
            txt_ids = torch.zeros(batch_size_cur, prompt_embeds.shape[1], 4, device=device, dtype=dtype)

            # Predict velocity with LoRA transformer
            pred_velocity = transformer(
                hidden_states=latents_t_packed,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]

            # Compute Flow Matching Loss
            loss = F.mse_loss(pred_velocity.float(), target_velocity.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            global_step += 1

            pbar.set_postfix(loss=f"{loss_val:.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint per epoch
        ckpt_dir = output_path / f"epoch_{epoch+1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        transformer.save_pretrained(ckpt_dir / "transformer_lora")
        save_adapter(adapter, ckpt_dir / "adapter.safetensors")

    # Save final
    final_dir = output_path / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    transformer.save_pretrained(final_dir / "transformer_lora")
    save_adapter(adapter, final_dir / "adapter.safetensors")

    elapsed = time.time() - start_time
    print(f"\n=== Training Completed in {elapsed/60:.2f} mins ===")
    print(f"LoRA & Adapter saved to: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FLUX.2 LoRA + Adapter")
    add_config_argument(parser)
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to pre-trained adapter weights")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory of training images + text captions")
    parser.add_argument("--hf-dataset", type=str, default=None, help="Hugging Face dataset name (e.g. jpawan33/kag100-image-captioning-dataset)")
    parser.add_argument("--synthetic-dir", type=str, default=None, help="Directory of synthetic teacher latents")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for LoRA checkpoints")
    args = parser.parse_args()

    train_lora(
        config=args.config,
        adapter_path=args.adapter_path,
        image_dir=args.image_dir,
        hf_dataset=args.hf_dataset,
        synthetic_dir=args.synthetic_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        img_size=args.img_size,
        output_dir=args.output_dir,
    )
