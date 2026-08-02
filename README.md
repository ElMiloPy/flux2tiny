# flux2tiny 🚀

**Replace a 4B-parameter text encoder with a 1B one — and teach the diffusion model to not notice.**

flux2tiny adapts [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) to use [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) (1.08B params) as text encoder instead of the original Qwen3-4B (4B params). The adaptation is done through a **three-stage knowledge distillation pipeline** that progressively transfers the teacher's behavior to the student configuration.

---

## How It Works

The full distillation pipeline has **three stages**, each building on the last:

### Stage 1 — Adapter Pre-training (Text Embedding Alignment)

> **Goal:** Teach a lightweight projection adapter to translate MiniCPM5-1B hidden states into the same embedding space as Qwen3-4B.

Both text encoders are frozen. For each training caption:
1. Run it through **Qwen3-4B** (teacher) and extract hidden states from layers `[8, 18, 28]`
2. Run it through **MiniCPM5-1B** (student) and extract hidden states from layers `[5, 12, 19]`
3. Concatenate the 3 teacher layers → `[batch, seq_len, 7680]` (3 × 2560)
4. Pass the 3 student layers through the **PerLayerProjection adapter** (3 independent `nn.Linear(1536→2560)` projections, concatenated) → `[batch, seq_len, 7680]`
5. Minimize **MSE loss** between adapter output and teacher target

The adapter is only **~11.8M parameters** — it trains in minutes on a single GPU.

```bash
python train_adapter.py --num-epochs 5 --batch-size 4
```

### Stage 2 — Synthetic Teacher Latent Generation

> **Goal:** Capture the full teacher pipeline's visual behavior as pre-encoded latent tensors for efficient downstream training.

Using the **original** FLUX.2-klein-4B pipeline (Qwen3-4B + FLUX.2 Transformer + VAE), generate raw bfloat16 latent tensors by running inference with `output_type="latent"` (bypasses the slow VAE decode → PNG encode → PNG decode cycle).

Each sample stores:
- The **prompt** (from [Gustavosta/Stable-Diffusion-Prompts](https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts))
- The **denoised latent tensor** (what the teacher's diffusion process produces)
- The **seed** (for reproducibility)

This runs at **~1.3 sec/sample** on an RTX A4500 and produces a compact `manifest.json` + `.pt` file per sample.

```bash
python generate_synthetic_dataset.py \
    --prompt-dataset Gustavosta/Stable-Diffusion-Prompts \
    --num-samples 15000 \
    --output-dir synthetic_sd_15k
```

### Stage 3 — Flow Matching LoRA Distillation

> **Goal:** Fine-tune the FLUX.2 Transformer backbone (via LoRA) and the projection adapter jointly, so the student pipeline reproduces the teacher's denoising behavior.

This is where the magic happens. For each training step:
1. Load a **(prompt, teacher_latent)** pair from Stage 2
2. Encode the prompt through **MiniCPM5-1B → Adapter** to get `prompt_embeds`
3. Sample a random timestep `t ∈ (0, 1)` and random noise `ε`
4. Interpolate: `z_t = (1 - t) · z_teacher + t · ε` (Flow Matching schedule)
5. The transformer predicts the velocity field `v_θ(z_t, t, prompt_embeds)`
6. Target velocity: `v* = ε - z_teacher`
7. Minimize **MSE loss** between `v_θ` and `v*`

Only the **LoRA weights** (~1.9M params, rank 16, targeting `to_q`, `to_k`, `to_v`, `to_out.0`) and the **adapter** (~11.8M params) are trainable. The FLUX.2 Transformer base weights (3.88B) and VAE remain frozen.

```bash
python train_lora.py \
    --synthetic-dir synthetic_sd_15k \
    --num-epochs 3 \
    --learning-rate 1e-4 \
    --output-dir lora_checkpoints
```

---

## Architecture Diagram

```
FLUX.2-klein-4B (Original)              flux2tiny (Adapted)
┌──────────────────────┐                ┌──────────────────────┐
│ Qwen3-4B (2560 dim)  │                │ MiniCPM5-1B (1536 d) │
│ 36 layers, 4B params │       →→→      │ 24 layers, 1B params │
│ hidden_size = 2560   │                │ hidden_size = 1536   │
└──────────┬───────────┘                └──────────┬───────────┘
           │ 3 × 2560 = 7680                       │ 3 × 1536 = 4608
           │                                       ▼
           │                            ┌────────────────────┐
           │                            │ Projection Adapter │
           │                            │ 4608 → 7680        │
           │                            │ ~11.8M params      │
           │                            └──────────┬─────────┘
           ▼                                       ▼
┌────────────────────────────────────────────────────────────┐
│    Flux2Transformer2DModel (3.88B) + PEFT LoRA (1.9M)     │
│    joint_attention_dim = 7680                              │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────┐
│       AutoencoderKLFlux2 (FLUX.2-small-decoder)           │
│       ~28M params decoder                                 │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
                          PIL Image
```

---

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/ElMiloPy/flux2tiny.git
cd flux2tiny

# Option A: automated conda setup
bash setup_env.sh
conda activate flux2tiny

# Option B: manual pip install
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Stage 1: Pre-train adapter (~5 min)
python train_adapter.py --num-epochs 5 --batch-size 4

# Stage 2: Generate teacher latents (~5.5 hours for 15k samples)
python generate_synthetic_dataset.py \
    --prompt-dataset Gustavosta/Stable-Diffusion-Prompts \
    --num-samples 15000 --output-dir synthetic_sd_15k

# Stage 3: LoRA distillation (~30 min for 3 epochs on 15k samples)
python train_lora.py \
    --synthetic-dir synthetic_sd_15k \
    --num-epochs 3 --output-dir lora_checkpoints
```

### 3. Generate Images

```bash
python generate.py "A photo of a dog sitting in a lush garden" \
    --adapter lora_checkpoints/final/adapter.safetensors \
    --lora lora_checkpoints/final/transformer_lora \
    --size 512x512 --seed 42
```

### 4. Compare Original vs. Adapted

```bash
python compare.py --prompt "A cat on a windowsill at sunset" --size 512x512
```

---

## Repository Structure

```
flux2tiny/
├── adapter.py                    # Projection adapter (PerLayerProjection / ConcatProjection)
├── pipeline.py                   # Flux2TinyPipeline — inference wrapper
├── generate.py                   # CLI image generation
├── compare.py                    # Side-by-side benchmark (original vs. adapted)
├── train_adapter.py              # Stage 1: Adapter pre-training (text embedding alignment)
├── generate_synthetic_dataset.py # Stage 2: Teacher latent generation
├── train_lora.py                 # Stage 3: Flow Matching LoRA distillation
├── setup_env.sh                  # Conda environment installer
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md
```

---

## Hardware Requirements

| Stage | Script | Peak VRAM | System RAM |
|:------|:-------|:----------|:-----------|
| 1. Adapter Pre-training | `train_adapter.py` | ~12 GB | ~24 GB |
| 2. Teacher Latent Generation | `generate_synthetic_dataset.py` | ~12 GB | ~24 GB |
| 3. LoRA Distillation | `train_lora.py` | ~10.3 GB | ~16 GB |
| Inference | `generate.py` | ~10 GB | ~16 GB |

*Tested on NVIDIA RTX A4500 Laptop GPU (16 GB VRAM), 64 GB System RAM, Linux Debian 13.*

---

## Dependencies

- Python ≥ 3.11
- PyTorch ≥ 2.4 (CUDA 12.4)
- [diffusers](https://github.com/huggingface/diffusers) (latest from git — required for `Flux2KleinPipeline`)
- [transformers](https://github.com/huggingface/transformers) (latest from git)
- [PEFT](https://github.com/huggingface/peft) (for LoRA)
- [accelerate](https://github.com/huggingface/accelerate), [safetensors](https://github.com/huggingface/safetensors), [datasets](https://github.com/huggingface/datasets)

---

## Upstream Model Licenses

| Model | License |
|:------|:--------|
| [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Apache 2.0 |
| [FLUX.2-small-decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder) | Apache 2.0 |
| [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | Apache 2.0 |

## License

This project's source code is released under the [MIT License](LICENSE).
