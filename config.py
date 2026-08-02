"""
flux2tiny — Model Configuration Registry.

Supports configurable student text encoders for knowledge distillation to FLUX.2-klein-4B:
  - MiniCPM5-1B (1.08B)
  - Qwen3.5-0.8B (0.8B)
  - Qwen3.5-2B (2.0B)
  - LFM2.5-230M (230M)
  - LFM2.5-350M (350M)
  - SmolLM2-135M-Instruct (135M)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from pathlib import Path
import argparse


@dataclass
class StudentConfig:
    name: str
    student_model_id: str
    hidden_size: int
    extract_layers: List[int]
    num_layers: int = 3
    teacher_model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    teacher_hidden_size: int = 2560
    teacher_extract_layers: List[int] = field(default_factory=lambda: [8, 18, 28])
    vae_model_id: str = "black-forest-labs/FLUX.2-small-decoder"
    default_adapter_dir: str = "adapter_checkpoints"
    default_lora_dir: str = "lora_checkpoints"
    description: str = ""

    @property
    def total_source_dim(self) -> int:
        """Total dimension of concatenated extracted student layers."""
        return self.hidden_size * self.num_layers

    @property
    def total_target_dim(self) -> int:
        """Total target joint attention dimension expected by FLUX.2 (7680)."""
        return self.teacher_hidden_size * len(self.teacher_extract_layers)

    def get_adapter_path(self, checkpoint: str = "adapter_best.safetensors") -> str:
        """Get adapter path with fallback to existing directory paths."""
        p1 = Path(self.default_adapter_dir) / checkpoint
        if p1.exists():
            return str(p1)
        p2 = Path("adapter_checkpoints") / checkpoint
        if p2.exists():
            return str(p2)
        return str(p1)

    def get_lora_path(self, checkpoint: str = "final/transformer_lora") -> str:
        """Get LoRA path with fallback to existing directory paths."""
        p1 = Path(self.default_lora_dir) / checkpoint
        if p1.exists():
            return str(p1)
        p2 = Path("lora_checkpoints_15k") / checkpoint
        if p2.exists():
            return str(p2)
        p3 = Path("lora_checkpoints") / checkpoint
        if p3.exists():
            return str(p3)
        return str(p1)


# Pre-defined student model presets
PRESETS: Dict[str, StudentConfig] = {
    "minicpm5-1b": StudentConfig(
        name="minicpm5-1b",
        student_model_id="openbmb/MiniCPM5-1B",
        hidden_size=1536,
        extract_layers=[5, 12, 19],
        default_adapter_dir="adapter_checkpoints/minicpm5_1b",
        default_lora_dir="lora_checkpoints/minicpm5_1b",
        description="MiniCPM5-1B (1.08B params, hidden_size=1536)",
    ),
    "minicpm-1b": StudentConfig(
        name="minicpm5-1b",
        student_model_id="openbmb/MiniCPM5-1B",
        hidden_size=1536,
        extract_layers=[5, 12, 19],
        default_adapter_dir="adapter_checkpoints/minicpm5_1b",
        default_lora_dir="lora_checkpoints/minicpm5_1b",
        description="MiniCPM5-1B (1.08B params, hidden_size=1536)",
    ),
    "qwen3.5-0.8b": StudentConfig(
        name="qwen3.5-0.8b",
        student_model_id="Qwen/Qwen3.5-0.8B",
        hidden_size=1024,
        extract_layers=[7, 15, 23],  # Full-attention layers in hybrid architecture
        default_adapter_dir="adapter_checkpoints/qwen3.5_0.8b",
        default_lora_dir="lora_checkpoints/qwen3.5_0.8b",
        description="Qwen3.5-0.8B (0.8B params, hidden_size=1024, hybrid linear/full attention)",
    ),
    "qwen3.5-2b": StudentConfig(
        name="qwen3.5-2b",
        student_model_id="Qwen/Qwen3.5-2B",
        hidden_size=2048,
        extract_layers=[7, 15, 23],  # Full-attention layers in hybrid architecture
        default_adapter_dir="adapter_checkpoints/qwen3.5_2b",
        default_lora_dir="lora_checkpoints/qwen3.5_2b",
        description="Qwen3.5-2B (2.0B params, hidden_size=2048, hybrid linear/full attention)",
    ),
    "lfm2.5-230m": StudentConfig(
        name="lfm2.5-230m",
        student_model_id="LiquidAI/LFM2.5-230M",
        hidden_size=1024,
        extract_layers=[3, 7, 11],  # 14 layers total
        default_adapter_dir="adapter_checkpoints/lfm2.5_230m",
        default_lora_dir="lora_checkpoints/lfm2.5_230m",
        description="Liquid AI LFM2.5-230M (230M params, hidden_size=1024)",
    ),
    "lfm2.5-350m": StudentConfig(
        name="lfm2.5-350m",
        student_model_id="LiquidAI/LFM2.5-350M",
        hidden_size=1024,
        extract_layers=[4, 8, 12],  # 16 layers total
        default_adapter_dir="adapter_checkpoints/lfm2.5_350m",
        default_lora_dir="lora_checkpoints/lfm2.5_350m",
        description="Liquid AI LFM2.5-350M (350M params, hidden_size=1024)",
    ),
    "smollm2-135m": StudentConfig(
        name="smollm2-135m",
        student_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        hidden_size=576,
        extract_layers=[8, 15, 23],  # 30 layers total
        default_adapter_dir="adapter_checkpoints/smollm2_135m",
        default_lora_dir="lora_checkpoints/smollm2_135m",
        description="SmolLM2-135M-Instruct (135M params, hidden_size=576)",
    ),
    "smollm2-135m-instruct": StudentConfig(
        name="smollm2-135m",
        student_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        hidden_size=576,
        extract_layers=[8, 15, 23],  # 30 layers total
        default_adapter_dir="adapter_checkpoints/smollm2_135m",
        default_lora_dir="lora_checkpoints/smollm2_135m",
        description="SmolLM2-135M-Instruct (135M params, hidden_size=576)",
    ),
}

DEFAULT_PRESET = "minicpm5-1b"


def get_student_config(preset_name: str = DEFAULT_PRESET) -> StudentConfig:
    """
    Retrieve a StudentConfig by preset name.
    If preset_name is not in PRESETS, fallback or raise ValueError.
    """
    key = preset_name.lower().strip()
    if key in PRESETS:
        return PRESETS[key]

    # Check if user provided an alias or direct model ID
    for name, cfg in PRESETS.items():
        if key == cfg.student_model_id.lower():
            return cfg

    raise ValueError(
        f"Unknown student config preset: '{preset_name}'. "
        f"Available presets: {list(PRESETS.keys())}"
    )


def add_config_argument(parser: argparse.ArgumentParser):
    """Utility helper to add --config CLI flag to scripts."""
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_PRESET,
        choices=list(PRESETS.keys()),
        help=f"Student model configuration preset (default: {DEFAULT_PRESET}). Options: {', '.join(PRESETS.keys())}",
    )
