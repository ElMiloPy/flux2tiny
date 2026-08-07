# flux2tiny 🚀

**Replace a 4B-parameter text encoder with a compact model — and teach the diffusion model to not notice.**

flux2tiny adapts [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) to use smaller text encoders instead of the default 4B-parameter Qwen3-4B. Adaptation is done through a **three-stage knowledge distillation pipeline**.

---

## Model Configurations

Configs live in `configs/*.json`. All scripts accept `--config configs/<name>.json`:

| Config | Model | Params | Hidden | Layers | Adapter |
|:-------|:------|:-------|:-------|:-------|:--------|
| `minicpm5-1b.json` | `openbmb/MiniCPM5-1B` | 1.08B | 1536 | `[5, 12, 19]` | ~11.8M |
| `qwen3.5-0.8b.json` | `Qwen/Qwen3.5-0.8B` | 0.8B | 1024 | `[7, 15, 23]` | ~7.9M |
| `qwen3.5-2b.json` | `Qwen/Qwen3.5-2B` | 2.0B | 2048 | `[7, 15, 23]` | ~15.7M |
| `qwen3.5-4b.json` | `Qwen/Qwen3.5-4B` | 4.0B | 2560 | `[7, 19, 31]` | ~19.7M |
| `lfm2.5-230m.json` | `LiquidAI/LFM2.5-230M` | 230M | 1024 | `[3, 7, 11]` | ~7.9M |
| `lfm2.5-350m.json` | `LiquidAI/LFM2.5-350M` | 350M | 1024 | `[4, 8, 12]` | ~7.9M |
| `gemma3-270m.json` | `google/gemma-3-270m-it` | 270M | 640 | `[5, 11, 17]` | ~4.9M |
| `smollm2-135m.json` | `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M | 576 | `[8, 15, 23]` | ~4.4M |

---

## How It Works

### Stage 1 — Teacher Latent & Embedding Generation

Generate synthetic latent-prompt pairs **alongside 1-to-1 precomputed teacher text embeddings** from the original FLUX.2 pipeline:

```bash
python generate_synthetic_dataset.py --num-samples 15000 --output-dir synthetic_sd_15k
```

### Stage 2 — Adapter Pre-training

Train a lightweight projection adapter to align student hidden states with precomputed teacher hidden states via MSE loss.

**Fast mode (using precomputed embeddings dataset):**
```bash
python train_adapter.py --config configs/minicpm5-1b.json --dataset-dir synthetic_sd_15k --num-epochs 5
```
*Note: Using `--dataset-dir` skips loading the heavy Qwen3-4B teacher model entirely during adapter training, providing >400 it/s throughput and requiring only ~2.5 GB VRAM.*

**Live mode (extracting teacher hidden states on-the-fly):**
```bash
python train_adapter.py --config configs/minicpm5-1b.json --num-epochs 5
```

### Stage 3 — Flow Matching LoRA Distillation

Joint training of adapter + FLUX.2 Transformer LoRA on teacher latents:

```bash
python train_lora.py --config configs/minicpm5-1b.json --synthetic-dir synthetic_sd_15k --num-epochs 3
```

---

## Architecture

```
Student Text Encoder (135M–4B)          Original: Qwen3-4B (4B)
  → extract 3 hidden layers
  → PerLayerProjection adapter (3 × Linear)
  → [B, seq, 7680] prompt_embeds
  → Flux2Transformer2DModel + LoRA (3.88B + 1.9M)
  → AutoencoderKLFlux2 decoder
  → PIL Image
```

---

## Quick Start

```bash
git clone https://github.com/ElMiloPy/flux2tiny.git
cd flux2tiny
bash setup_env.sh
conda activate flux2tiny
```

### Generate Images

```bash
python generate.py "A photo of a dog in a garden" --config configs/minicpm5-1b.json --size 512x512 --seed 42
```

### Upload Weights

```bash
python upload_to_hub.py --config configs/minicpm5-1b.json --repo-id Emilio407/flux2tiny-weights
```

---

## Repository Structure

```
flux2tiny/
├── configs/                          # Student model JSON configs
│   ├── minicpm5-1b.json
│   ├── qwen3.5-{0.8b,2b,4b}.json
│   ├── lfm2.5-{230m,350m}.json
│   ├── gemma3-270m.json
│   └── smollm2-135m.json
├── config.py                         # Config loader & hardware helpers
├── adapter.py                        # Projection adapter module
├── pipeline.py                       # Inference pipeline
├── generate.py                       # CLI image generation
├── compare.py                        # Side-by-side benchmark
├── generate_synthetic_dataset.py     # Stage 1: Teacher latent & embedding generation
├── train_adapter.py                  # Stage 2: Adapter pre-training
├── train_lora.py                     # Stage 3: LoRA distillation
├── upload_to_hub.py                  # HuggingFace Hub upload
├── setup_env.sh                      # Conda environment setup
├── requirements.txt
├── LICENSE                           # MIT
└── README.md
```

---

## Hardware Requirements

| Stage | Script | Peak VRAM |
|:------|:-------|:----------|
| 1. Synthetic Dataset | `generate_synthetic_dataset.py` | ~12 GB |
| 2. Adapter (Precomputed) | `train_adapter.py --dataset-dir ...` | ~2.5 GB |
| 2. Adapter (Live Teacher) | `train_adapter.py` | ~12 GB (or ~2.5 GB with `unsloth/Qwen3-4B-bnb-4bit`) |
| 3. LoRA | `train_lora.py` | ~10 GB |
| Inference | `generate.py` | ~4 GB (with CPU offload) |

---

## License

Source code released under the [MIT License](LICENSE).

