"""
flux2tiny — Adapter pre-training (Stage 1).

Knowledge distillation: trains a projection adapter to align student text
encoder hidden states with the Qwen3-4B teacher's hidden states via MSE loss.

Both encoders are frozen — only the lightweight adapter trains.

Usage:
  python train_adapter.py --config configs/minicpm5-1b.json --num-epochs 5

Multi-GPU (dual GTX 1080):
  accelerate launch --multi_gpu train_adapter.py --config configs/lfm2.5-230m.json
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from accelerate import Accelerator

from adapter import PerLayerProjection, ConcatProjection, save_adapter
from config import get_student_config, add_config_argument, get_default_dtype


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
SAMPLE_PROMPTS = [
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


class CaptionDataset(Dataset):
    def __init__(self, captions: list[str] | None = None, num_samples: int = 5000):
        if captions:
            self.captions = captions
        else:
            self.captions = [
                f"{SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]} (variant {i // len(SAMPLE_PROMPTS)})"
                for i in range(num_samples)
            ]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cosine_lr(step: int, warmup: int, total: int, max_lr: float) -> float:
    if step < warmup:
        return max_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def extract_hidden_states(model, tokenizer, texts, layers, max_len, device, dtype):
    """Extract hidden states from specific layers of a frozen LM."""
    # For models loaded with device_map="auto", get the actual first-param device
    input_device = next(model.parameters()).device if str(device) == "auto" else device

    inputs = tokenizer(texts, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=max_len).to(input_device)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    return [outputs.hidden_states[i + 1].to(dtype) for i in layers]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(config: dict):
    accelerator = Accelerator(mixed_precision="fp16" if config.get("fp16") else "no")
    torch.manual_seed(config["seed"])

    device = accelerator.device
    dtype = torch.float16 if config.get("fp16") else get_default_dtype()
    output_dir = Path(config["output_dir"])

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "train_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"Device: {device} | Dtype: {dtype} | Processes: {accelerator.num_processes}")
        print(f"Output: {output_dir}")

    # --- Load teacher (frozen) ---
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if accelerator.is_main_process:
        print(f"\nLoading teacher: {config['teacher_model']}...")

    teacher_on_cpu = config.get("teacher_on_cpu", False)
    teacher_device = "cpu" if teacher_on_cpu else device

    teacher_tokenizer = AutoTokenizer.from_pretrained(config["teacher_model"], trust_remote_code=True)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    teacher_model = AutoModelForCausalLM.from_pretrained(
        config["teacher_model"], dtype=dtype,
        device_map=teacher_device, trust_remote_code=True,
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # --- Load student (frozen) ---
    if accelerator.is_main_process:
        print(f"Loading student: {config['student_model']}...")

    student_tokenizer = AutoTokenizer.from_pretrained(config["student_model"], trust_remote_code=True)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    student_model = AutoModelForCausalLM.from_pretrained(
        config["student_model"], dtype=dtype,
        device_map=device, trust_remote_code=True,
    )
    student_model.eval()
    for p in student_model.parameters():
        p.requires_grad = False

    # --- Create adapter ---
    if config.get("adapter_type") == "concat":
        adapter = ConcatProjection(
            source_dim=config["student_hidden_size"],
            target_total_dim=config["teacher_hidden_size"] * config["num_extract_layers"],
            num_layers=config["num_extract_layers"],
        )
    else:
        adapter = PerLayerProjection(
            source_dim=config["student_hidden_size"],
            target_dim_per_layer=config["teacher_hidden_size"],
            num_layers=config["num_extract_layers"],
        )
    adapter = adapter.to(device=device, dtype=torch.float32)

    if accelerator.is_main_process:
        print(f"Adapter: {adapter.total_params:,} params")

    # --- Data ---
    captions = None
    if config.get("captions_file") and Path(config["captions_file"]).exists():
        captions = Path(config["captions_file"]).read_text().strip().splitlines()
        if accelerator.is_main_process:
            print(f"Loaded {len(captions)} captions from {config['captions_file']}")

    dataset = CaptionDataset(captions=captions, num_samples=config["num_train_samples"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config["learning_rate"],
                                  weight_decay=config.get("weight_decay", 0.01))
    loss_fn = nn.MSELoss()

    adapter, optimizer, dataloader = accelerator.prepare(adapter, optimizer, dataloader)

    total_steps = len(dataloader) * config["num_epochs"]
    warmup = config.get("warmup_steps", 100)
    log_every = config.get("log_every", 50)
    save_every = config.get("save_every", 500)

    if accelerator.is_main_process:
        print(f"Dataset: {len(dataset)} captions | Steps: {total_steps}\n")

    # --- Training loop ---
    global_step = 0
    running_loss = 0.0
    best_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        epoch_loss, epoch_steps = 0.0, 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}",
                    disable=not accelerator.is_main_process)

        for batch_texts in pbar:
            lr = cosine_lr(global_step, warmup, total_steps, config["learning_rate"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Teacher forward
            teacher_hidden = extract_hidden_states(
                teacher_model, teacher_tokenizer, list(batch_texts),
                config["teacher_extract_layers"], config["max_seq_len"],
                device=teacher_device, dtype=dtype,
            )
            teacher_target = torch.cat(
                [h.to(device=device, dtype=torch.float32) for h in teacher_hidden], dim=-1
            )

            # Student forward
            student_hidden = extract_hidden_states(
                student_model, student_tokenizer, list(batch_texts),
                config["student_extract_layers"], config["max_seq_len"],
                device=device, dtype=dtype,
            )
            student_hidden = [h.to(device=device, dtype=torch.float32) for h in student_hidden]

            # Align sequence lengths (different tokenizers)
            min_seq = min(teacher_target.shape[1], student_hidden[0].shape[1])
            teacher_target = teacher_target[:, :min_seq, :]
            student_hidden = [h[:, :min_seq, :] for h in student_hidden]

            # Forward + backward
            output = adapter(student_hidden)
            loss = loss_fn(output, teacher_target)

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

                if global_step % log_every == 0:
                    avg = running_loss / log_every
                    print(f"  Step {global_step}/{total_steps} | loss={avg:.6f} | lr={lr:.2e}")
                    running_loss = 0.0

                if global_step % save_every == 0:
                    unwrapped = accelerator.unwrap_model(adapter)
                    save_adapter(unwrapped, output_dir / f"adapter_step{global_step}.safetensors")

        if accelerator.is_main_process:
            avg_epoch = epoch_loss / max(1, epoch_steps)
            print(f"--- Epoch {epoch+1} | Avg Loss: {avg_epoch:.6f} ---")

            if avg_epoch < best_loss:
                best_loss = avg_epoch
                save_adapter(accelerator.unwrap_model(adapter),
                             output_dir / "adapter_best.safetensors")

    if accelerator.is_main_process:
        save_adapter(accelerator.unwrap_model(adapter), output_dir / "adapter_final.safetensors")
        print(f"\n=== Training Complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULTS = {
    "teacher_model": "Qwen/Qwen3-4B",
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flux2tiny projection adapter (Stage 1)")
    add_config_argument(parser)
    parser.add_argument("--teacher-model", type=str, default=DEFAULTS["teacher_model"])
    parser.add_argument("--student-model", type=str, default=None)
    parser.add_argument("--adapter-type", type=str, default="per_layer", choices=["per_layer", "concat"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--num-epochs", type=int, default=DEFAULTS["num_epochs"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--max-seq-len", type=int, default=DEFAULTS["max_seq_len"])
    parser.add_argument("--num-train-samples", type=int, default=DEFAULTS["num_train_samples"])
    parser.add_argument("--captions-file", type=str, default=None, help="Text file with one caption per line")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--teacher-on-cpu", action="store_true")
    args = parser.parse_args()

    student_cfg = get_student_config(args.config)

    config = {**DEFAULTS}
    config.update({
        "preset": student_cfg.name,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model or student_cfg.student_model_id,
        "student_hidden_size": student_cfg.hidden_size,
        "student_extract_layers": student_cfg.extract_layers,
        "adapter_type": args.adapter_type,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "max_seq_len": args.max_seq_len,
        "num_train_samples": args.num_train_samples,
        "captions_file": args.captions_file,
        "output_dir": args.output_dir or student_cfg.default_adapter_dir,
        "seed": args.seed,
        "teacher_on_cpu": args.teacher_on_cpu,
        "fp16": args.fp16,
    })

    train(config)
