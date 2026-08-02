# flux2tiny 🚀

Adapts **[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)** to use **[MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)** (1.08B params) as text encoder instead of Qwen3-4B, fine-tuned via **Flow Matching LoRA Distillation** directly on pre-encoded teacher latents.

---

## 📐 Architecture

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
│      Flux2Transformer2DModel (3.88B) + PEFT LoRA (1.9M)     │
│      joint_attention_dim = 7680                            │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────┐
│         AutoencoderKLFlux2 (FLUX.2-small-decoder)          │
│         ~28M params decoder                                │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
                          PIL Image
```

---

## 🛠️ Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/flux2tiny.git
cd flux2tiny

# Setup conda environment (PyTorch CUDA 12.4 + diffusers + PEFT)
bash setup_env.sh
conda activate flux2tiny
```

---

## ⚡ Quick Start: Image Generation

Generate images using fine-tuned LoRA weights and projection adapter:

```bash
python generate.py "A photo of a dog sitting in a lush garden" \
    --adapter lora_checkpoints/final/adapter.safetensors \
    --lora lora_checkpoints/final/transformer_lora \
    --size 512x512 --seed 42
```

---

## 🎓 Teacher Latent Distillation Workflow

### Step 1: Generate Synthetic Teacher Latents
Generate raw bfloat16 latent tensors directly from original FLUX.2 (Qwen3-4B teacher + FLUX.2 Transformer) at **1.3s/sample** (bypasses slow PNG disk encoding/decoding):

```bash
# Generate 15,000 synthetic teacher latents from Hugging Face prompts
python generate_synthetic_dataset.py \
    --prompt-dataset Gustavosta/Stable-Diffusion-Prompts \
    --num-samples 15000 \
    --output-dir synthetic_sd_15k
```

### Step 2: Train Student LoRA + Adapter
Train the MiniCPM5-1B projection adapter AND FLUX.2 Transformer LoRA parameters using Flow Matching Loss ($v = z_1 - z_0$) directly on the teacher latents:

```bash
python train_lora.py \
    --synthetic-dir synthetic_sd_15k \
    --num-epochs 3 \
    --learning-rate 1e-4 \
    --output-dir lora_checkpoints_15k
```

---

## 📁 Repository Structure

```
flux2tiny/
├── adapter.py                   # Projection adapter module (1536 → 2560)
├── pipeline.py                  # Custom Flux2TinyPipeline wrapping Flux2KleinPipeline
├── generate.py                  # CLI image generation script
├── train_adapter.py             # Initial linear distillation script
├── train_lora.py                # Flow Matching LoRA training script
├── generate_synthetic_dataset.py# High-speed Teacher Latent Generator
├── compare.py                   # Benchmark & visual comparison script
├── setup_env.sh                 # Environment installer
├── requirements.txt             # Base requirements
└── README.md                    # Project documentation
```

---

## 💻 Hardware Requirements

| Phase | Peak VRAM | System RAM |
|:---|:---|:---|
| Synthetic Teacher Generation | ~12 GB | ~24 GB |
| Flow Matching LoRA Training | ~10.3 GB | ~16 GB |
| Image Generation | ~10 GB | ~16 GB |

*Tested on NVIDIA RTX A4500 Laptop GPU (16 GB VRAM), 64 GB System RAM, Linux Debian 13.*

---

## 📜 License

- **FLUX.2-klein-4B & FLUX.2-small-decoder**: Apache 2.0
- **MiniCPM5-1B**: Apache 2.0
- **flux2tiny Code**: Apache 2.0
