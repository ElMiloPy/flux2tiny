"""
flux2tiny — Model Configuration Registry.

Loads JSON student model configurations from the `configs/` directory.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union
from pathlib import Path
import argparse


CONFIGS_DIR = Path(__file__).parent / "configs"
DEFAULT_PRESET = "minicpm5-1b"


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


def list_available_configs() -> Dict[str, Path]:
    """Scan `configs/` directory for available JSON configurations."""
    configs = {}
    if CONFIGS_DIR.exists():
        for p in CONFIGS_DIR.glob("*.json"):
            configs[p.stem.lower()] = p
    return configs


def get_student_config(preset_or_path: str = DEFAULT_PRESET) -> StudentConfig:
    """
    Retrieve a StudentConfig by preset name, config name, or direct JSON filepath.
    """
    path_obj = Path(preset_or_path)

    # 1. Direct path to JSON file
    if path_obj.exists() and path_obj.is_file():
        return StudentConfig.from_json(path_obj)

    # 2. Preset name matching file in configs/ directory
    key = preset_or_path.lower().strip()
    if key.endswith(".json"):
        key = key[:-5]

    available = list_available_configs()

    if key in available:
        return StudentConfig.from_json(available[key])

    # Aliases
    aliases = {
        "minicpm-1b": "minicpm5-1b",
        "smollm2-135m-instruct": "smollm2-135m",
    }
    if key in aliases and aliases[key] in available:
        return StudentConfig.from_json(available[aliases[key]])

    # 3. Match against HF model IDs in available configs
    for cfg_path in available.values():
        cfg = StudentConfig.from_json(cfg_path)
        if key == cfg.student_model_id.lower():
            return cfg

    valid_names = sorted(list(available.keys()))
    raise ValueError(
        f"Unknown student config or file path: '{preset_or_path}'. "
        f"Available presets in configs/: {valid_names}"
    )


def add_config_argument(parser: argparse.ArgumentParser):
    """Utility helper to add --config CLI flag to scripts."""
    available_names = sorted(list(list_available_configs().keys()))
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_PRESET,
        help=f"Student model config preset or JSON filepath (default: {DEFAULT_PRESET}). Options: {', '.join(available_names)}",
    )
