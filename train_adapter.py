"""
flux2tiny — Training script for the projection adapter.

Knowledge distillation approach:
  1. Load both Qwen3-4B (teacher) and student text encoders
  2. For a batch of captions, extract hidden states from both
  3. Train the projection adapter to minimize MSE between
     adapter(student_hidden_states) and teacher_hidden_states
  4. All encoder weights are frozen — only the adapter trains

Supports single GPU, multi-GPU layer sharding, and accelerate DDP (dual GPU parallel).
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from adapter import PerLayerProjection, ConcatProjection, save_adapter
from config import get_student_config, add_config_argument, get_default_dtype
from accelerate import Accelerator


# ---------------------------------------------------------------------------
# Base Configuration Defaults
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "teacher_model": "Qwen/Qwen3-4B",
    "adapter_type": "per_layer",  # "per_layer" or "concat"
    "teacher_hidden_size": 2560,
    "teacher_extract_layers": [8, 18, 28],
    "num_extract_layers": 3,
    "max_seq_len": 128,
    "batch_size": 4,
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "num_epochs": 5,
    "warmup_steps": 100,
    "save_every": 500,
    "log_every": 50,
    "num_train_samples": 5000,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Simple Dataset of Synthetic Prompts
# ---------------------------------------------------------------------------
class CaptionDataset(Dataset):
    def __init__(self, captions: list[str] = None, num_samples: int = 5000):
        if captions is not None:
            self.captions = captions
        else:
            base_prompts = [
                "A photo of a dog sitting in a lush green garden",
                "A futuristic city with flying cars at sunset",
                "An oil painting of a cottage near a serene lake",
                "A close-up portrait of a woman with red hair",
                "A cute cat wearing a tiny wizard hat",
                "A watercolor landscape of snow-capped mountains",
                "A delicious slice of chocolate cake on a ceramic plate",
                "A retro 80s arcade with neon lights",
                "A majestic dragon perched atop a rocky peak",
                "A cozy coffee shop interior on a rainy afternoon",
                "A vibrant coral reef with colorful tropical fish",
                "A vintage sports car driving along a coastal highway",
                "An astronaut standing on the surface of Mars",
                "A whimsical treehouse in an enchanted forest",
                "A minimalist modern kitchen with marble countertops",
                "A dramatic stormy sky over a wheat field",
                "A robot reading a book in a dusty library",
                "A plate of fresh sushi with ginger and wasabi",
                "A mechanical pocket watch with visible gears",
                "A quiet cobblestone street in a European village",
            ]
            self.captions = [
                f"{base_prompts[i % len(base_prompts)]} (variant {i // len(base_prompts)})"
                for i in range(num_samples)
            ]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx]


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------
def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Hidden state extraction helpers
# ---------------------------------------------------------------------------
def extract_hidden_states(
    model: nn.Module,
    tokenizer,
    texts: list[str],
    extract_layers: list[int],
    max_seq_len: int,
    device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """
    Run text through a causal LM and extract hidden states at specified layers.

    Returns a list of tensors, one per layer, each [batch, seq_len, hidden_size].
    """
    input_device = next(model.parameters()).device if str(device) == "auto" else device

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    ).to(input_device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states

    extracted = []
    for layer_idx in extract_layers:
        hs = hidden_states[layer_idx + 1].to(dtype)
        extracted.append(hs)

    return extracted


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(config: dict):
    accelerator = Accelerator(
        mixed_precision="fp16" if config.get("fp16") else "no"
    )

    torch.manual_seed(config["seed"])

    device = accelerator.device
    dtype = torch.float16 if config.get("fp16", False) else get_default_dtype()
    use_multi_gpu_sharding = config.get("multi_gpu", False) and accelerator.num_processes == 1

    output_dir = Path(config["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"Device: {device} | Dtype: {dtype} | Processes: {accelerator.num_processes} | Sharding: {use_multi_gpu_sharding}")
        print(f"Output: {output_dir}")

    # -----------------------------------------------------------------------
    # Load teacher (Qwen3-4B) — frozen
    # -----------------------------------------------------------------------
    if config.get("teacher_on_cpu", False):
        teacher_device = "cpu"
    elif use_multi_gpu_sharding:
        teacher_device = "auto"
    else:
        teacher_device = device

    if accelerator.is_main_process:
        print(f"\nLoading teacher: {config['teacher_model']} on {teacher_device}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    teacher_tokenizer = AutoTokenizer.from_pretrained(
        config["teacher_model"], trust_remote_code=True
    )
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    teacher_model = AutoModelForCausalLM.from_pretrained(
        config["teacher_model"],
        torch_dtype=dtype,
        device_map=teacher_device,
        trust_remote_code=True,
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # -----------------------------------------------------------------------
    # Load student (MiniCPM5-1B / LFM2.5 / SmolLM2, etc.) — frozen
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        print(f"\nLoading student: {config['student_model']}...")
    student_tokenizer = AutoTokenizer.from_pretrained(
        config["student_model"], trust_remote_code=True
    )
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    student_model = AutoModelForCausalLM.from_pretrained(
        config["student_model"],
        torch_dtype=dtype,
        device_map=device if not use_multi_gpu_sharding else "auto",
        trust_remote_code=True,
    )
    student_model.eval()
    for p in student_model.parameters():
        p.requires_grad = False

    # -----------------------------------------------------------------------
    # Create adapter
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        print(f"\nCreating adapter (type={config['adapter_type']})...")
    if config["adapter_type"] == "per_layer":
        adapter = PerLayerProjection(
            source_dim=config["student_hidden_size"],
            target_dim_per_layer=config["teacher_hidden_size"],
            num_layers=config["num_extract_layers"],
        )
    else:
        adapter = ConcatProjection(
            source_dim=config["student_hidden_size"],
            target_total_dim=config["teacher_hidden_size"] * config["num_extract_layers"],
            num_layers=config["num_extract_layers"],
        )
    adapter = adapter.to(device=device, dtype=torch.float32)  # Train in fp32 for stability

    # -----------------------------------------------------------------------
    # Dataset and dataloader
    # -----------------------------------------------------------------------
    dataset = CaptionDataset(num_samples=config["num_train_samples"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    # -----------------------------------------------------------------------
    # Optimizer & Prepare
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = nn.MSELoss()

    adapter, optimizer, dataloader = accelerator.prepare(adapter, optimizer, dataloader)

    total_steps = len(dataloader) * config["num_epochs"]
    if accelerator.is_main_process:
        print(f"\nDataset: {len(dataset)} captions")
        print(f"Total training steps: {total_steps}")
        print("\n=== Training ===")

    global_step = 0
    running_loss = 0.0

    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{config['num_epochs']}",
            disable=not accelerator.is_main_process,
        )
        for batch_texts in pbar:
            # Update LR
            lr = get_lr(global_step, config["warmup_steps"], total_steps, config["learning_rate"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # --- Teacher forward ---
            teacher_hidden = extract_hidden_states(
                teacher_model, teacher_tokenizer, list(batch_texts),
                config["teacher_extract_layers"],
                config["max_seq_len"],
                device=teacher_device, dtype=dtype,
            )
            # Move teacher outputs to GPU and concatenate
            teacher_hidden = [h.to(device=device, dtype=torch.float32) for h in teacher_hidden]
            teacher_target = torch.cat(teacher_hidden, dim=-1)  # [B, seq_len, 7680]

            # --- Student forward (on GPU) ---
            student_hidden = extract_hidden_states(
                student_model, student_tokenizer, list(batch_texts),
                config["student_extract_layers"],
                config["max_seq_len"],
                device=device, dtype=dtype,
            )
            student_hidden = [h.to(device=device, dtype=torch.float32) for h in student_hidden]

            # Handle potential seq_len mismatch (different tokenizers)
            min_seq = min(teacher_target.shape[1], student_hidden[0].shape[1])
            teacher_target = teacher_target[:, :min_seq, :]
            student_hidden = [h[:, :min_seq, :] for h in student_hidden]

            # --- Adapter forward ---
            adapter_output = adapter(student_hidden)  # [B, seq_len, 7680]

            # --- Loss and update ---
            loss = loss_fn(adapter_output, teacher_target)

            optimizer.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            running_loss += loss_val
            epoch_loss += loss_val
            epoch_steps += 1
            global_step += 1

            if accelerator.is_main_process:
                pbar.set_postfix(loss=f"{loss_val:.6f}", lr=f"{lr:.2e}")

                if global_step % config["log_every"] == 0:
                    avg_loss = running_loss / config["log_every"]
                    print(f"  Step {global_step}/{total_steps} | loss={avg_loss:.6f} | lr={lr:.2e}")
                    running_loss = 0.0

                if global_step % config["save_every"] == 0:
                    unwrapped_adapter = accelerator.unwrap_model(adapter)
                    save_adapter(
                        unwrapped_adapter,
                        output_dir / f"adapter_step{global_step}.safetensors",
                    )

        if accelerator.is_main_process:
            avg_epoch_loss = epoch_loss / max(1, epoch_steps)
            print(f"--- Epoch {epoch+1} Complete | Average Loss: {avg_epoch_loss:.6f} ---")

    if accelerator.is_main_process:
        unwrapped_adapter = accelerator.unwrap_model(adapter)
        final_adapter_path = output_dir / "adapter_final.safetensors"
        best_adapter_path = output_dir / "adapter_best.safetensors"
        save_adapter(unwrapped_adapter, final_adapter_path)
        save_adapter(unwrapped_adapter, best_adapter_path)

        print(f"\n=== Training Complete ===")
        print(f"Adapter saved to: {final_adapter_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flux2tiny projection adapter")
    add_config_argument(parser)
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_CONFIG["teacher_model"])
    parser.add_argument("--student-model", type=str, default=None, help="Override student model ID")
    parser.add_argument("--adapter-type", type=str, default="per_layer", choices=["per_layer", "concat"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_CONFIG["max_seq_len"])
    parser.add_argument("--num-train-samples", type=int, default=DEFAULT_CONFIG["num_train_samples"])
    parser.add_argument("--captions-file", type=str, default=None, help="Text file with one caption per line")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for adapter checkpoints")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--teacher-on-cpu", action="store_true", help="Force teacher model to run on CPU")
    args = parser.parse_args()

    student_cfg = get_student_config(args.config)
    student_model_id = args.student_model or student_cfg.student_model_id
    output_dir = args.output_dir or student_cfg.default_adapter_dir

    config = DEFAULT_CONFIG.copy()
    config.update({
        "preset": student_cfg.name,
        "teacher_model": args.teacher_model,
        "student_model": student_model_id,
        "student_hidden_size": student_cfg.hidden_size,
        "student_extract_layers": student_cfg.extract_layers,
        "adapter_type": args.adapter_type,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "max_seq_len": args.max_seq_len,
        "num_train_samples": args.num_train_samples,
        "output_dir": output_dir,
        "seed": args.seed,
        "teacher_on_cpu": args.teacher_on_cpu,
        "multi_gpu": args.multi_gpu,
        "fp16": args.fp16,
    })

    train(config)
