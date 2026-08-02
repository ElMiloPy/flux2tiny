# flux2tiny 🚀

**Replace a 4B-parameter text encoder with a compact model — and teach the diffusion model to not notice.**

flux2tiny adapts [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) to use smaller text encoders instead of the default 4B-parameter Qwen3-4B text encoder. Adaptation is done through a **three-stage knowledge distillation pipeline** that progressively transfers the teacher's behavior to the student configuration.

---

## ⚙️ Model Configurations (`configs/*.json`)

Model configurations are stored as clean, human-readable JSON files in the [`configs/`](file:///home/emilio/Documents/flux2tiny/configs) directory.

All scripts accept either a JSON file path (e.g. `--config configs/minicpm5-1b.json`) or a config name in `configs/`:

| Config File | Model ID | Params | Hidden Size | Layers Extracted | Adapter Size |
|:------------|:---------|:-------|:------------|:-----------------|:-------------|
| [`configs/minicpm5-1b.json`](file:///home/emilio/Documents/flux2tiny/configs/minicpm5-1b.json) | `openbmb/MiniCPM5-1B` | 1.08B | 1536 | `[5, 12, 19]` | ~11.8M params |
| [`configs/qwen3.5-0.8b.json`](file:///home/emilio/Documents/flux2tiny/configs/qwen3.5-0.8b.json) | `Qwen/Qwen3.5-0.8B` | 0.8B | 1024 | `[7, 15, 23]` | ~7.9M params |
| [`configs/qwen3.5-2b.json`](file:///home/emilio/Documents/flux2tiny/configs/qwen3.5-2b.json) | `Qwen/Qwen3.5-2B` | 2.0B | 2048 | `[7, 15, 23]` | ~15.7M params |
| [`configs/qwen3.5-4b.json`](file:///home/emilio/Documents/flux2tiny/configs/qwen3.5-4b.json) | `Qwen/Qwen3.5-4B` | 4.0B | 2560 | `[7, 19, 31]` | ~19.7M params |
| [`configs/lfm2.5-230m.json`](file:///home/emilio/Documents/flux2tiny/configs/lfm2.5-230m.json) | `LiquidAI/LFM2.5-230M` | 230M | 1024 | `[3, 7, 11]` | ~7.9M params |
| [`configs/lfm2.5-350m.json`](file:///home/emilio/Documents/flux2tiny/configs/lfm2.5-350m.json) | `LiquidAI/LFM2.5-350M` | 350M | 1024 | `[4, 8, 12]` | ~7.9M params |
| [`configs/smollm2-135m.json`](file:///home/emilio/Documents/flux2tiny/configs/smollm2-135m.json) | `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M | 576 | `[8, 15, 23]` | ~4.4M params |

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
python train_adapter.py --config configs/minicpm5-1b.json --num-epochs 5

# Train adapter for Qwen3.5-4B
python train_adapter.py --config configs/qwen3.5-4b.json --num-epochs 5

# Train adapter using a custom JSON config file
python train_adapter.py --config configs/my_custom_model.json --num-epochs 5
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
python train_lora.py --config configs/minicpm5-1b.json --synthetic-dir synthetic_sd_15k --num-epochs 3

# Flow Matching LoRA distillation for Qwen3.5-4B
python train_lora.py --config configs/qwen3.5-4b.json --synthetic-dir synthetic_sd_15k --num-epochs 3
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
    --config configs/minicpm5-1b.json --size 512x512 --seed 42
```

Generate images with Qwen3.5-4B:
```bash
python generate.py "A photo of a dog sitting in a lush garden" \
    --config configs/qwen3.5-4b.json --size 512x512 --seed 42
```

### 3. Upload Trained Weights to Hugging Face

```bash
python upload_to_hub.py --config configs/minicpm5-1b.json --repo-id Emilio407/flux2tiny-weights
python upload_to_hub.py --config configs/qwen3.5-4b.json --repo-id Emilio407/flux2tiny-qwen3.5-4b-weights
```

---

## Repository Structure

```
flux2tiny/
├── configs/                      # JSON configuration directory for student models
│   ├── minicpm5-1b.json
│   ├── qwen3.5-0.8b.json
│   ├── qwen3.5-2b.json
│   ├── qwen3.5-4b.json
│   ├── lfm2.5-230m.json
│   ├── lfm2.5-350m.json
│   └── smollm2-135m.json
├── config.py                     # Central configuration module & loader
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
