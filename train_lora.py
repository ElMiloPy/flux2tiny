"""
flux2tiny — LoRA distillation training (Stage 3).

Jointly trains a PEFT LoRA on the FLUX.2 Transformer + the projection adapter
using Flow Matching Loss on teacher latents (from Stage 2) or real images.

Supports pre-cached student text embeddings, multi-worker data loading,
gradient accumulation, and pre-allocated grid position IDs for maximum throughput.

Usage:
  python train_lora.py --config configs/minicpm5-1b.json --synthetic-dir synthetic_sd_15k --batch-size 4 --num-workers 4
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from accelerate import Accelerator

from adapter import save_adapter, load_adapter
from config import get_student_config, add_config_argument, StudentConfig, get_default_dtype


# ---------------------------------------------------------------------------
# Pre-caching Helper
# ---------------------------------------------------------------------------
def precache_student_embeddings(
    synthetic_dir: Path,
    preset_name: str,
    student_model_id: str,
    student_extract_layers: list[int],
    device: torch.device,
    dtype: torch.dtype,
    max_seq_len: int = 128,
):
    """Pre-compute and save student hidden states to disk for zero-overhead training."""
    cache_dir = synthetic_dir / preset_name
    manifest_file = synthetic_dir / "manifest.json"
    if not manifest_file.exists():
        return None

    with open(manifest_file) as f:
        manifest = json.load(f)

    all_exist = cache_dir.exists() and all(
        (cache_dir / f"embed_{item['id']:06d}.pt").exists() or (cache_dir / f"student_embed_{item['id']:06d}.pt").exists()
        for item in manifest
    )
    if all_exist:
        print(f"--> Found pre-cached student embeddings in {cache_dir.name}")
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Pre-caching student embeddings for {preset_name} ({student_model_id}) ===")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(student_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_encoder = AutoModelForCausalLM.from_pretrained(
        student_model_id, dtype=dtype, trust_remote_code=True
    ).to(device)
    text_encoder.eval()

    with torch.no_grad():
        for item in tqdm(manifest, desc=f"Pre-caching {preset_name} embeds"):
            idx = item["id"]
            save_path = cache_dir / f"embed_{idx:06d}.pt"
            if save_path.exists():
                continue
            prompt = item["prompt"]
            inputs = tokenizer(
                prompt, return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_seq_len,
            ).to(device)
            outputs = text_encoder(**inputs, output_hidden_states=True, return_dict=True)
            hidden_list = [outputs.hidden_states[i + 1].cpu() for i in student_extract_layers]
            torch.save(hidden_list, save_path)

    del text_encoder
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Completed pre-caching {len(manifest)} items to {cache_dir}\n")
    return cache_dir


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ImageCaptionDataset(Dataset):
    """Loads synthetic latents, precomputed student embeds, HF datasets, or local images."""

    def __init__(self, image_dir=None, hf_dataset=None, synthetic_dir=None,
                 student_embeds_dir=None, img_size=512):
        self.img_size = img_size
        self.samples = []
        self.hf_data = None
        self.synthetic_dir = None
        self.student_embeds_dir = Path(student_embeds_dir) if student_embeds_dir else None
        self.synthetic_manifest = []

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        if synthetic_dir and Path(synthetic_dir).exists():
            manifest_file = Path(synthetic_dir) / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file) as f:
                    self.synthetic_manifest = json.load(f)
                self.synthetic_dir = Path(synthetic_dir)
                print(f"Loaded {len(self.synthetic_manifest)} synthetic latents from {synthetic_dir}")
                return

        if hf_dataset:
            from datasets import load_dataset
            print(f"Loading HF dataset: {hf_dataset}...")
            self.hf_data = load_dataset(hf_dataset, split="train")
            print(f"Loaded {len(self.hf_data)} samples")
            return

        if image_dir and Path(image_dir).exists():
            for p in Path(image_dir).iterdir():
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    txt = p.with_suffix(".txt")
                    caption = txt.read_text().strip() if txt.exists() else p.stem.replace("_", " ")
                    self.samples.append((str(p), caption))
            print(f"Loaded {len(self.samples)} image-caption pairs from {image_dir}")
            return

        raise ValueError("Must provide a valid --synthetic-dir, --hf-dataset, or --image-dir for LoRA training.")

    def __len__(self):
        if self.synthetic_dir:
            return len(self.synthetic_manifest)
        if self.hf_data is not None:
            return len(self.hf_data)
        if self.samples:
            return len(self.samples)
        return 0

    def __getitem__(self, idx):
        if self.synthetic_dir:
            item = self.synthetic_manifest[idx]
            latent = torch.load(self.synthetic_dir / item["latent_file"], map_location="cpu", weights_only=True)
            sample = {"latent": latent, "caption": item["prompt"]}
            if self.student_embeds_dir:
                embed_path = self.student_embeds_dir / f"embed_{item['id']:06d}.pt"
                if not embed_path.exists():
                    embed_path = self.student_embeds_dir / f"student_embed_{item['id']:06d}.pt"
                if embed_path.exists():
                    student_hidden = torch.load(embed_path, map_location="cpu", weights_only=True)
                    sample["student_hidden"] = student_hidden
            return sample

        if self.hf_data is not None:
            item = self.hf_data[idx]
            img = item["image"].convert("RGB")
            caption = item.get("text", item.get("caption", "a photo"))
        elif self.samples:
            path, caption = self.samples[idx]
            img = Image.open(path).convert("RGB")
        else:
            raise RuntimeError("Invalid dataset state.")

        return {"pixel_values": self.transform(img), "caption": caption}


def collate_fn(batch):
    result = {}
    captions = [item["caption"] for item in batch]
    result["caption"] = captions

    if "latent" in batch[0]:
        result["latent"] = torch.stack([item["latent"] for item in batch])

    if "pixel_values" in batch[0]:
        result["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])

    if "student_hidden" in batch[0] and batch[0]["student_hidden"] is not None:
        num_layers = len(batch[0]["student_hidden"])
        student_hidden = []
        for l in range(num_layers):
            layer_tensors = torch.stack([item["student_hidden"][l].squeeze(0) for item in batch])
            student_hidden.append(layer_tensors)
        result["student_hidden"] = student_hidden

    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_lora(
    config: str | StudentConfig = "configs/minicpm5-1b.json",
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
    gradient_accumulation_steps: int = 1,
    num_workers: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    img_size: int = 512,
    fp16: bool = False,
):
    accelerator = Accelerator(
        mixed_precision="fp16" if fp16 else "no",
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    device = accelerator.device
    dtype = torch.float16 if fp16 else get_default_dtype()

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
    output_path = Path(output_dir)

    if accelerator.is_main_process:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"=== Training FLUX.2 LoRA + Adapter ({student_cfg.name}) ===")
        print(f"  Device: {device} | Dtype: {dtype} | Processes: {accelerator.num_processes}")
        print(f"  Image: {img_size}x{img_size} | LoRA rank: {lora_rank} | Batch size: {batch_size} (Accum: {gradient_accumulation_steps})")

    # 1. Check pre-cached student embeddings (folder named by preset_name e.g. smollm2-135m)
    student_embeds_dir = None
    if synthetic_dir and Path(synthetic_dir).exists():
        if accelerator.is_main_process:
            student_embeds_dir = precache_student_embeddings(
                synthetic_dir=Path(synthetic_dir),
                preset_name=student_cfg.name,
                student_model_id=student_model_id,
                student_extract_layers=student_extract_layers,
                device=device, dtype=dtype,
            )
        accelerator.wait_for_everyone()
        student_embeds_dir = Path(synthetic_dir) / student_cfg.name

    use_precached_embeds = student_embeds_dir and student_embeds_dir.exists()
    text_encoder, tokenizer = None, None

    if not use_precached_embeds:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if accelerator.is_main_process:
            print(f"Loading student text encoder: {student_model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(student_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        text_encoder = AutoModelForCausalLM.from_pretrained(
            student_model_id, dtype=dtype, trust_remote_code=True
        ).to(device)
        text_encoder.eval()
        for p in text_encoder.parameters():
            p.requires_grad = False
    else:
        if accelerator.is_main_process:
            print(f"--> Using pre-cached student embeddings from {student_embeds_dir.name} (skipping text encoder VRAM load)")

    # 2. Adapter — trainable
    if accelerator.is_main_process:
        print(f"  Adapter: {adapter_path}")

    adapter = load_adapter(
        adapter_path, adapter_type="per_layer",
        source_dim=student_cfg.hidden_size,
        target_dim=student_cfg.teacher_hidden_size,
        num_layers=student_cfg.num_layers,
        device=str(device), dtype=torch.float32,
    )
    adapter.train()
    for p in adapter.parameters():
        p.requires_grad = True

    # 3. VAE — frozen
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(vae_model_id, torch_dtype=dtype).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # 4. Transformer + LoRA
    from diffusers.models import Flux2Transformer2DModel
    from peft import LoraConfig, get_peft_model

    transformer = Flux2Transformer2DModel.from_pretrained(
        flux_model_id, subfolder="transformer", torch_dtype=dtype
    ).to(device)

    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()

    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    transformer = get_peft_model(transformer, lora_config)

    if accelerator.is_main_process:
        transformer.print_trainable_parameters()

    # 5. Dataset & DataLoader
    dataset = ImageCaptionDataset(
        image_dir=image_dir, hf_dataset=hf_dataset,
        synthetic_dir=synthetic_dir, student_embeds_dir=student_embeds_dir,
        img_size=img_size,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers if use_precached_embeds else 0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0 if use_precached_embeds else False,
    )

    # 6. Optimizer
    trainable_params = list(adapter.parameters()) + list(transformer.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    adapter, transformer, optimizer, dataloader = accelerator.prepare(
        adapter, transformer, optimizer, dataloader
    )

    # Pre-compute spatial grid position IDs
    h_ids, w_ids = img_size // 16, img_size // 16
    grid = torch.cartesian_prod(
        torch.arange(1, device=device),
        torch.arange(h_ids, device=device),
        torch.arange(w_ids, device=device),
        torch.arange(1, device=device),
    ).to(dtype)
    cached_img_ids = grid.unsqueeze(0)  # [1, h_ids*w_ids, 4]

    # Auto-resume
    start_epoch = 0
    from safetensors.torch import load_file
    epoch_dirs = [d for d in output_path.glob("epoch_*") if d.is_dir() and (d / "adapter.safetensors").exists()]
    if epoch_dirs:
        latest_dir = max(epoch_dirs, key=lambda d: int(d.name.replace("epoch_", "")))
        start_epoch = int(latest_dir.name.replace("epoch_", ""))
        adapter_state = load_file(str(latest_dir / "adapter.safetensors"))
        accelerator.unwrap_model(adapter).load_state_dict(adapter_state)
        if accelerator.is_main_process:
            print(f"--> Auto-resuming LoRA training from epoch {start_epoch} ({latest_dir.name})")

    global_step = start_epoch * len(dataloader)
    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}",
                    disable=not accelerator.is_main_process)

        for batch in pbar:
            with accelerator.accumulate(transformer):
                captions = batch["caption"]

                if "student_hidden" in batch:
                    student_hidden = [h.to(device=device, dtype=torch.float32) for h in batch["student_hidden"]]
                    prompt_embeds = adapter(student_hidden).to(dtype=dtype)
                else:
                    inputs = tokenizer(
                        captions, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=128,
                    ).to(device)

                    with torch.no_grad():
                        outputs = text_encoder(**inputs, output_hidden_states=True, return_dict=True)
                        hidden_list = [outputs.hidden_states[i + 1] for i in student_extract_layers]

                    prompt_embeds = adapter([h.float() for h in hidden_list]).to(dtype=dtype)

                # Latents
                if "latent" in batch:
                    latents_0 = batch["latent"].to(device=device, dtype=dtype)
                    if latents_0.ndim == 3:
                        latents_0 = latents_0.unsqueeze(0)
                else:
                    pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
                    with torch.no_grad():
                        latents_0 = vae.encode(pixel_values).latent_dist.sample()

                # Pack latents for FLUX.2
                B, C, H, W = latents_0.shape

                def pack(tensor):
                    x = tensor.view(B, C, H // 2, 2, W // 2, 2)
                    return x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, -1).permute(0, 2, 1)

                latents_packed = pack(latents_0)

                # Flow Matching
                t = torch.rand(B, device=device, dtype=dtype)
                noise = pack(torch.randn_like(latents_0))
                t_exp = t.view(B, 1, 1)

                latents_t = (1.0 - t_exp) * latents_packed + t_exp * noise
                target_velocity = noise - latents_packed

                img_ids = cached_img_ids.expand(B, -1, -1)
                txt_ids = torch.zeros(B, prompt_embeds.shape[1], 4, device=device, dtype=dtype)

                # Predict velocity
                pred = transformer(
                    hidden_states=latents_t, timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=txt_ids, img_ids=img_ids,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(pred.float(), target_velocity.float())

                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                global_step += 1

                if accelerator.is_main_process:
                    pbar.set_postfix(loss=f"{loss_val:.4f}")

        if accelerator.is_main_process:
            avg = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} | Avg Loss: {avg:.4f}")

            # Save checkpoint
            ckpt_dir = output_path / f"epoch_{epoch+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            unwrapped_transformer = accelerator.unwrap_model(transformer)
            unwrapped_transformer.save_pretrained(ckpt_dir / "transformer_lora")
            save_adapter(accelerator.unwrap_model(adapter), ckpt_dir / "adapter.safetensors")

    # Final save
    if accelerator.is_main_process:
        final_dir = output_path / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(transformer).save_pretrained(final_dir / "transformer_lora")
        save_adapter(accelerator.unwrap_model(adapter), final_dir / "adapter.safetensors")

        elapsed = time.time() - start_time
        print(f"\n=== Training Complete ({elapsed/60:.1f} min) ===")
        print(f"Saved to: {final_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FLUX.2 LoRA + Adapter (Stage 3)")
    add_config_argument(parser)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--hf-dataset", type=str, default=None)
    parser.add_argument("--synthetic-dir", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    train_lora(
        config=args.config,
        adapter_path=args.adapter_path,
        image_dir=args.image_dir,
        hf_dataset=args.hf_dataset,
        synthetic_dir=args.synthetic_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        img_size=args.img_size,
        output_dir=args.output_dir,
        fp16=args.fp16,
    )
