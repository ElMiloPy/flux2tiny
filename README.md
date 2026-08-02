# flux2tiny 🚀

**Replace a 4B-parameter text encoder with a 1B (or 0.8B) model — and teach the diffusion model to not notice.**

flux2tiny adapts [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) to use smaller text encoders instead of the default 4B-parameter Qwen3-4B text encoder. Adaptation is done through a **three-stage knowledge distillation pipeline** that progressively transfers the teacher's behavior to the student configuration.

Supported student model configurations:
- `minicpm-1b` — [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) (1.08B params, `hidden_size=1536`)
- `qwen3.5-0.8b` — [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (0.8B params, `hidden_size=1024`, hybrid attention)

---

## ⚙️ Model Configurations (`config.py`)

All scripts in `flux2tiny` accept the `--config` CLI flag to select the student text encoder:

| Preset Name | Model ID | Hidden Size | Layers Extracted | Adapter Size |
|:------------|:---------|:------------|:-----------------|:-------------|
| `minicpm-1b` | `openbmb/MiniCPM5-1B` | 1536 | `[5, 12, 19]` | ~11.8M params |
| `qwen3.5-0.8b` | `Qwen/Qwen3.5-0.8B` | 1024 | `[7, 15, 23]` | ~7.9M params |

---

## How It Works

The full distillation pipeline has **three stages**, each building on the last:

### Stage 1 — Adapter Pre-training (Text Embedding Alignment)

> **Goal:** Teach a lightweight projection adapter to translate student hidden states into the same embedding space as Qwen3-4B.

Both text encoders are frozen. For each training caption:
1. Run it through **Qwen3-4B** (teacher) and extract hidden states from layers `[8, 18, 28]` → concatenate to `[batch, seq_len, 7680]` (3 × 2560)
2. Run it through **Student Model** (MiniCPM5-1B or Qwen3.5-0.8B) and extract layers
3. Pass student layers through **PerLayerProjection adapter** (3 independent `nn.Linear(student_dim → 2560)` projections, concatenated)
4. Minimize **MSE loss** between adapter output and teacher target

```bash
# Train adapter for MiniCPM5-1B
python train_adapter.py --config minicpm-1b --num-epochs 5

# Train adapter for Qwen3.5-0.8B
python train_adapter.py --config qwen3.5-0.8b --num-epochs 5
```

### Stage 2 — Synthetic Teacher Latent Generation

> **Goal:** Capture the full teacher pipeline's visual behavior as pre-encoded latent tensors for efficient downstream training.

Using the **original** FLUX.2-klein-4B pipeline (Qwen3-4B + FLUX.2 Transformer + VAE), generate raw bfloat16 latent tensors directly:

```bash
python generate_synthetic_dataset.py \
    --prompt-dataset Gustavosta/Stable-Diffusion-Prompts \
    --num-samples 15000 \
    --output-dir synthetic_sd_15k
```

### Stage 3 — Flow Matching LoRA Distillation

> **Goal:** Fine-tune the FLUX.2 Transformer backbone (via LoRA) and the projection adapter jointly on teacher latents.

```bash
# Flow Matching LoRA distillation for MiniCPM5-1B
python train_lora.py \
    --config minicpm-1b \
    --synthetic-dir synthetic_sd_15k \
    --num-epochs 3

# Flow Matching LoRA distillation for Qwen3.5-0.8B
python train_lora.py \
    --config qwen3.5-0.8b \
    --synthetic-dir synthetic_sd_15k \
    --num-epochs 3
```

---

## Architecture Diagram

```
FLUX.2-klein-4B (Original)              flux2tiny (Adapted)
┌──────────────────────┐                ┌──────────────────────────────┐
│ Qwen3-4B (2560 dim)  │                │ Student Model (1024/1536 dim)│
│ 36 layers, 4B params │       →→→      │ MiniCPM5-1B or Qwen3.5-0.8B  │
│ hidden_size = 2560   │                │ 24 layers, ~0.8B-1B params   │
└──────────┬───────────┘                └──────────────┬───────────────┘
           │ 3 × 2560 = 7680                                   │ 3 × D = 3D
           │                                                   ▼
           │                            ┌──────────────────────────────┐
           │                            │ Projection Adapter           │
           │                            │ 3D → 7680 (3 × 2560)         │
           │                            │ ~7.9M - 11.8M params         │
           │                            └──────────────┬───────────────┘
           ▼                                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│        Flux2Transformer2DModel (3.88B) + PEFT LoRA (1.9M)            │
│        joint_attention_dim = 7680                                    │
└─────────────────────────────┬────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│           AutoencoderKLFlux2 (FLUX.2-small-decoder)                  │
│           ~28M params decoder                                        │
└─────────────────────────────┬────────────────────────────────────────┘
                              ▼
                          PIL Image
```

---

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/ElMiloPy/flux2tiny.git
cd flux2tiny

# Setup conda env
bash setup_env.sh
conda activate flux2tiny
```

### 2. Generate Images

Generate images with MiniCPM5-1B:
```bash
python generate.py "A photo of a dog sitting in a lush garden" \
    --config minicpm-1b --size 512x512 --seed 42
```

Generate images with Qwen3.5-0.8B:
```bash
python generate.py "A photo of a dog sitting in a lush garden" \
    --config qwen3.5-0.8b --size 512x512 --seed 42
```

### 3. Upload Trained Weights to Hugging Face

```bash
python upload_to_hub.py --config minicpm-1b --repo-id Emilio407/flux2tiny-weights
python upload_to_hub.py --config qwen3.5-0.8b --repo-id Emilio407/flux2tiny-qwen3.5-0.8b-weights
```

---

## Repository Structure

```
flux2tiny/
├── config.py                     # Central configuration registry (MiniCPM5-1B, Qwen3.5-0.8B)
├── adapter.py                    # Projection adapter (PerLayerProjection / ConcatProjection)
├── pipeline.py                   # Flux2TinyPipeline — inference wrapper
├── generate.py                   # CLI image generation
├── compare.py                    # Side-by-side benchmark (original vs. adapted)
├── train_adapter.py              # Stage 1: Adapter pre-training (text embedding alignment)
├── generate_synthetic_dataset.py # Stage 2: Teacher latent generation
├── train_lora.py                 # Stage 3: Flow Matching LoRA distillation
├── upload_to_hub.py              # Hugging Face Hub upload utility
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

## Upstream Model Licenses

| Model | License |
|:------|:--------|
| [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Apache 2.0 |
| [FLUX.2-small-decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder) | Apache 2.0 |
| [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | Apache 2.0 |
| [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Apache 2.0 |

## License

This project's source code is released under the [MIT License](LICENSE).
