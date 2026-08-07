"""
flux2tiny — Projection adapter module.

Bridges the dimensional gap between a student text encoder and
FLUX.2-klein-4B's transformer (joint_attention_dim=7680).

FLUX.2 extracts hidden states from 3 layers of its text encoder and
concatenates them: 3 × 2560 = 7680. We project the student's smaller
hidden states to match this target dimension.

Two strategies:
  1. PerLayerProjection — independent linear per extracted layer (default)
  2. ConcatProjection   — concatenate all layers, single linear
"""

import torch
import torch.nn as nn
from pathlib import Path
from safetensors.torch import save_file, load_file


class PerLayerProjection(nn.Module):
    """Projects each extracted hidden-state layer with LayerNorm + 2-layer MLP (SiLU), then concatenates."""

    def __init__(self, source_dim: int, target_dim_per_layer: int = 2560, num_layers: int = 3, bias: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(source_dim),
                nn.Linear(source_dim, target_dim_per_layer, bias=bias),
                nn.SiLU(),
                nn.Linear(target_dim_per_layer, target_dim_per_layer, bias=bias),
            )
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states_list: list[torch.Tensor]) -> torch.Tensor:
        """[B, seq, source_dim] × num_layers → [B, seq, target_dim_per_layer × num_layers]"""
        assert len(hidden_states_list) == self.num_layers
        return torch.cat([proj(hs) for proj, hs in zip(self.projections, hidden_states_list)], dim=-1)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ConcatProjection(nn.Module):
    """Concatenates all layers first, then projects with a single linear."""

    def __init__(self, source_dim: int, target_total_dim: int = 7680, num_layers: int = 3, bias: bool = True):
        super().__init__()
        self.projection = nn.Linear(source_dim * num_layers, target_total_dim, bias=bias)

    def forward(self, hidden_states_list: list[torch.Tensor]) -> torch.Tensor:
        """[B, seq, source_dim] × num_layers → [B, seq, target_total_dim]"""
        return self.projection(torch.cat(hidden_states_list, dim=-1))

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def save_adapter(adapter: nn.Module, path: str | Path):
    """Save adapter weights as safetensors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(adapter.state_dict(), str(path))
    print(f"Saved adapter ({adapter.total_params:,} params) → {path}")


def load_adapter(
    path: str | Path,
    adapter_type: str = "per_layer",
    source_dim: int = 1536,
    target_dim: int = 2560,
    num_layers: int = 3,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """Load adapter weights from safetensors."""
    if adapter_type == "per_layer":
        adapter = PerLayerProjection(source_dim, target_dim, num_layers)
    elif adapter_type == "concat":
        adapter = ConcatProjection(source_dim, target_dim * num_layers, num_layers)
    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}")

    state_dict = load_file(str(path))
    adapter.load_state_dict(state_dict)
    adapter = adapter.to(device=device, dtype=dtype)
    adapter.eval()
    print(f"Loaded adapter ({adapter.total_params:,} params) ← {path}")
    return adapter


if __name__ == "__main__":
    print("=== Adapter Sanity Check ===")
    B, S = 2, 77

    for src_dim, name in [(1536, "MiniCPM5-1B"), (1024, "Qwen3.5-0.8B"), (576, "SmolLM2-135M")]:
        dummy = [torch.randn(B, S, src_dim) for _ in range(3)]

        pl = PerLayerProjection(src_dim, 2560, 3)
        out = pl(dummy)
        assert out.shape == (B, S, 7680), f"PerLayer failed for {name}"

        cc = ConcatProjection(src_dim, 7680, 3)
        out2 = cc(dummy)
        assert out2.shape == (B, S, 7680), f"Concat failed for {name}"

        print(f"  {name} (dim={src_dim}): PerLayer={pl.total_params:,}p, Concat={cc.total_params:,}p ✓")

    print("All checks passed ✓")
