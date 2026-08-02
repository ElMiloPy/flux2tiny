# flux2tiny 🚀

**Replace a 4B-parameter text encoder with a compact model — and teach the diffusion model to not notice.**

flux2tiny adapts [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) to use smaller text encoders instead of the default 4B-parameter Qwen3-4B text encoder. Adaptation is done through a **three-stage knowledge distillation pipeline** that progressively transfers the teacher's behavior to the student configuration.

---

## ⚙️ Model Configurations (`config.py`)

All scripts in `flux2tiny` accept the `--config` CLI flag to select the student text encoder:

| Preset Name | Model ID | Params | Hidden Size | Layers Extracted | Adapter Size |
|:------------|:---------|:-------|:------------|:-----------------|:-------------|
| `minicpm5-1b` | `openbmb/MiniCPM5-1B` | 1.08B | 1536 | `[5, 12, 19]` | ~11.8M params |
| `qwen3.5-0.8b` | `Qwen/Qwen3.5-0.8B` | 0.8B | 1024 | `[7, 15, 23]` | ~7.9M params |
| `qwen3.5-2b` | `Qwen/Qwen3.5-2B` | 2.0B | 2048 | `[7, 15, 23]` | ~15.7M params |
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B` | 4.0B | 2560 | `[7, 19, 31]` | ~19.7M params |
| `lfm2.5-230m` | `LiquidAI/LFM2.5-230M` | 230M | 1024 | `[3, 7, 11]` | ~7.9M params |
| `lfm2.5-350m` | `LiquidAI/LFM2.5-350M` | 350M | 1024 | `[4, 8, 12]` | ~7.9M params |
| `smollm2-135m` | `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M | 576 | `[8, 15, 23]` | ~4.4M params |

---

## How It Works

The full distillation pipeline has **three stages**, each building on the last:

### Stage 1 — Adapter Pre-training (Text Embedding Alignment)

> **Goal:** Teach a lightweight projection adapter to translate student hidden states into the same embedding space as Qwen3-4B.

Both text encoders are frozen. For each training caption:
1. Run it through **Qwen3-4B** (teacher) and extract hidden states from layers `[8, 18, 28]` → concatenate to `[batch, seq_len, 7680]` (3 × 2560)
2. Run it through **Student Model** (MiniCPM, Qwen3.5, LFM2.5, SmolLM2, etc.) and extract 3 layers
3. Pass student layers through **PerLayerProjection adapter** (3 independent `nn.Linear(student_dim → 2560)` projections, concatenated)
4. Minimize **MSE loss** between adapter output and teacher target

```bash
# Train adapter for MiniCPM5-1B
python train_adapter.py --config minicpm5-1b --num-epochs 5

# Train adapter for Qwen3.5-4B
python train_adapter.py --config qwen3.5-4b --num-epochs 5

# Train adapter for SmolLM2-135M
python train_adapter.py --config smollm2-135m --num-epochs 5
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
python train_lora.py --config minicpm5-1b --synthetic-dir synthetic_sd_15k --num-epochs 3

# Flow Matching LoRA distillation for Qwen3.5-4B
python train_lora.py --config qwen3.5-4b --synthetic-dir synthetic_sd_15k --num-epochs 3
```

---

## Architecture Diagram

```
FLUX.2-klein-4B (Original)              flux2tiny (Adapted)
┌──────────────────────┐                ┌──────────────────────────────┐
│ Qwen3-4B (2560 dim)  │                │ Student Model (576-2560 dim) │
│ 36 layers, 4B params │       →→→      │ MiniCPM, Qwen3.5, LFM, Smol  │
│ hidden_size = 2560   │                │ 135M - 4.0B params           │
└──────────┬───────────┘                └──────────────┬───────────────┘
           │ 3 × 2560 = 7680                                   │ 3 × D = 3D
           │                                                   ▼
           │                            ┌──────────────────────────────┐
           │                            │ Projection Adapter           │
           │                            │ 3D → 7680 (3 × 2560)         │
           │                            │ ~4.4M - 19.7M params         │
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
    --config minicpm5-1b --size 512x512 --seed 42
```

Generate images with Qwen3.5-4B:
```bash
python generate.py "A photo of a dog sitting in a lush garden" \
    --config qwen3.5-4b --size 512x512 --seed 42
```

### 3. Upload Trained Weights to Hugging Face

```bash
python upload_to_hub.py --config minicpm5-1b --repo-id Emilio407/flux2tiny-weights
python upload_to_hub.py --config qwen3.5-4b --repo-id Emilio407/flux2tiny-qwen3.5-4b-weights
```

---

## Repository Structure

```
flux2tiny/
├── config.py                     # Central configuration registry
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

## License

- Source code released under the [MIT License](LICENSE).
