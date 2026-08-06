"""
flux2tiny — Model configuration & hardware helpers.

Loads student model configs from JSON files in configs/.
Auto-detects GPU dtype (fp16 for Pascal / bf16 for Ampere+).
"""

import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Union
from pathlib import Path

import torch

CONFIGS_DIR = Path(__file__).parent / "configs"
DEFAULT_CONFIG = str(CONFIGS_DIR / "minicpm5-1b.json")


def get_default_dtype() -> torch.dtype:
    """Auto-detect: fp16 for Pascal (GTX 1080), bf16 for Ampere+."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
        return torch.float16
    return torch.bfloat16


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
    def total_target_dim(self) -> int:
        """joint_attention_dim expected by FLUX.2 (3 × 2560 = 7680)."""
        return self.teacher_hidden_size * len(self.teacher_extract_layers)

    def get_adapter_path(self, filename: str = "adapter_best.safetensors") -> str:
        p = Path(self.default_adapter_dir) / filename
        if p.exists():
            return str(p)
        fallback = Path("adapter_checkpoints") / filename
        return str(fallback if fallback.exists() else p)

    def get_lora_path(self, filename: str = "final/transformer_lora") -> str:
        p = Path(self.default_lora_dir) / filename
        if p.exists():
            return str(p)
        fallback = Path("lora_checkpoints") / filename
        return str(fallback if fallback.exists() else p)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "StudentConfig":
        with open(path, "r") as f:
            return cls(**json.load(f))

    def to_json(self, path: Union[str, Path]):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def get_student_config(name_or_path: str = DEFAULT_CONFIG) -> StudentConfig:
    """Load a StudentConfig from a JSON path or config name."""
    p = Path(name_or_path)

    # Direct path
    if p.exists() and p.is_file():
        return StudentConfig.from_json(p)

    # Lookup in configs/
    candidate = CONFIGS_DIR / (f"{name_or_path}.json" if not name_or_path.endswith(".json") else p.name)
    if candidate.exists():
        return StudentConfig.from_json(candidate)

    raise FileNotFoundError(f"Config not found: '{name_or_path}' or '{candidate}'")


def add_config_argument(parser: argparse.ArgumentParser):
    """Add --config, --fp16 CLI flags."""
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help=f"Path to student JSON config (default: {DEFAULT_CONFIG})")
    parser.add_argument("--fp16", action="store_true",
                        help="Force float16 (auto-detected for Pascal GPUs)")
