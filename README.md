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

Generate synthetic latent-prompt pairs along with precomputed teacher embeddings from the original FLUX.2 pipeline:

```bash
python generate_synthetic_dataset.py --num-samples 15000 --output-dir synthetic_sd_15k
```

### Stage 2 — Adapter Pre-training

Train a lightweight projection adapter using precomputed teacher embeddings from the synthetic dataset (fast & low VRAM), or live teacher extraction:

```bash
# Recommended: Train from precomputed embeds dataset
python train_adapter.py --config configs/minicpm5-1b.json --dataset-dir synthetic_sd_15k --num-epochs 5

# Alternative: Live teacher extraction
python train_adapter.py --config configs/minicpm5-1b.json --num-epochs 5
```

### Stage 3 — Flow Matching LoRA Distillation

Joint training of adapter + FLUX.2 Transformer LoRA on teacher latents (with automatic student embedding pre-caching, multi-worker DataLoader, and gradient accumulation):

```bash
python train_lora.py --config configs/minicpm5-1b.json --synthetic-dir synthetic_sd_15k --batch-size 4 --num-workers 4 --num-epochs 3
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
├── train_adapter.py                  # Stage 1: Adapter pre-training
├── generate_synthetic_dataset.py     # Stage 2: Teacher latent generation
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
| 1. Latents & Embeds | `generate_synthetic_dataset.py` | ~12 GB |
| 2. Adapter | `train_adapter.py` | ~2.5 GB (with `--dataset-dir`) or ~12 GB (live teacher) |
| 3. LoRA | `train_lora.py` | ~10 GB |
| Inference | `generate.py` | ~4 GB (with CPU offload) |

---

## License

Source code released under the [MIT License](LICENSE).
