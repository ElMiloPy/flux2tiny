"""
flux2tiny — Model Configuration Registry & Hardware Helpers.

Loads JSON student model configurations directly from `.json` file paths.
Auto-detects GPU capabilities (e.g. GTX 1080 Pascal fp16 vs. Ampere/Ada bf16)
and configures multi-GPU layer sharding across available graphics cards.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Union, Optional
from pathlib import Path
import argparse
import torch


CONFIGS_DIR = Path(__file__).parent / "configs"
DEFAULT_CONFIG_PATH = str(CONFIGS_DIR / "minicpm5-1b.json")


def get_default_dtype() -> torch.dtype:
    """
    Auto-detect optimal dtype based on GPU architecture.
    GTX 1080 / Pascal GPUs (Compute Capability < 8.0) do not natively support
    hardware bfloat16, so we default to float16 (fp16) for performance and compatibility.
    Ampere / Ada / Hopper GPUs (Compute Capability >= 8.0) use bfloat16.
    """
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if cap[0] < 8:
            return torch.float16
    return torch.bfloat16


def get_device_map(multi_gpu: bool = False) -> Optional[Union[str, dict]]:
    """
    Return 'auto' device_map for model layer sharding across multiple GPUs
    if multi_gpu is True or if multiple CUDA devices are available.
    """
    if multi_gpu or (torch.cuda.is_available() and torch.cuda.device_count() > 1):
        return "auto"
    return None


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
        """Get adapter path with fallback to root adapter_checkpoints/."""
        p1 = Path(self.default_adapter_dir) / checkpoint
        if p1.exists():
            return str(p1)
        p2 = Path("adapter_checkpoints") / checkpoint
        if p2.exists():
            return str(p2)
        return str(p1)

    def get_lora_path(self, checkpoint: str = "final/transformer_lora") -> str:
        """Get LoRA path with fallback to root lora_checkpoints/."""
        p1 = Path(self.default_lora_dir) / checkpoint
        if p1.exists():
            return str(p1)
        p2 = Path("lora_checkpoints") / checkpoint
        if p2.exists():
            return str(p2)
        return str(p1)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "StudentConfig":
        """Load StudentConfig from a JSON file."""
        path = Path(json_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, json_path: Union[str, Path]):
        """Save StudentConfig to a JSON file."""
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


def get_student_config(preset_or_path: str = DEFAULT_CONFIG_PATH) -> StudentConfig:
    """
    Retrieve a StudentConfig by JSON file path or name in configs/.
    """
    path_obj = Path(preset_or_path)

    # 1. Direct path to existing JSON file
    if path_obj.exists() and path_obj.is_file():
        return StudentConfig.from_json(path_obj)

    # 2. Check under configs/ directory
    if not preset_or_path.endswith(".json"):
        candidate = CONFIGS_DIR / f"{preset_or_path}.json"
    else:
        candidate = CONFIGS_DIR / path_obj.name

    if candidate.exists():
        return StudentConfig.from_json(candidate)

    raise FileNotFoundError(
        f"Config file not found at '{preset_or_path}' or '{candidate}'."
    )


def add_config_argument(parser: argparse.ArgumentParser):
    """Utility helper to add --config and hardware CLI flags to scripts."""
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to student JSON config file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Enable multi-GPU layer sharding across all available GPUs (e.g. 2x GTX 1080)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Force float16 precision (recommended for GTX 1080 / Pascal GPUs)",
    )
