"""
flux2tiny — Projection adapter module.

Bridges the dimensional gap between MiniCPM5-1B (hidden_size=1536) and
FLUX.2-klein-4B's transformer (joint_attention_dim=7680).

The FLUX.2 pipeline extracts hidden states from 3 layers of the text encoder
and concatenates them: 3 × hidden_size = joint_attention_dim.
  - Original: 3 × 2560 (Qwen3-4B) = 7680
  - Ours:     3 × 1536 (MiniCPM5-1B) = 4608 → project to 7680

We provide two adapter strategies:
  1. PerLayerProjection: independent linear projection per extracted layer
  2. ConcatProjection: concatenate all layers then project (fewer params but less flexible)
"""

import torch
import torch.nn as nn
from pathlib import Path
from safetensors.torch import save_file, load_file


class PerLayerProjection(nn.Module):
    """
    Projects each extracted hidden-state layer independently.

    For 3 layers of MiniCPM5-1B (dim=1536 each), this creates 3 separate
    nn.Linear(1536, 2560) projections, then concatenates to get 7680.
    This preserves the per-layer structure the transformer expects.
    """

    def __init__(
        self,
        source_dim: int = 1536,
        target_dim_per_layer: int = 2560,
        num_layers: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.source_dim = source_dim
        self.target_dim_per_layer = target_dim_per_layer
        self.projections = nn.ModuleList([
            nn.Linear(source_dim, target_dim_per_layer, bias=bias)
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hidden_states_list: List of 3 tensors, each [batch, seq_len, 1536]
        Returns:
            Tensor of shape [batch, seq_len, 7680]
        """
        assert len(hidden_states_list) == self.num_layers, (
            f"Expected {self.num_layers} hidden state tensors, got {len(hidden_states_list)}"
        )
        projected = [proj(hs) for proj, hs in zip(self.projections, hidden_states_list)]
        return torch.cat(projected, dim=-1)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ConcatProjection(nn.Module):
    """
    Concatenates all extracted layers then projects with a single linear.

    3 × 1536 = 4608 → nn.Linear(4608, 7680)
    Simpler, fewer parameters, but doesn't preserve per-layer structure.
    """

    def __init__(
        self,
        source_dim: int = 1536,
        target_total_dim: int = 7680,
        num_layers: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.concat_dim = source_dim * num_layers
        self.projection = nn.Linear(self.concat_dim, target_total_dim, bias=bias)

    def forward(self, hidden_states_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hidden_states_list: List of 3 tensors, each [batch, seq_len, 1536]
        Returns:
            Tensor of shape [batch, seq_len, 7680]
        """
        concatenated = torch.cat(hidden_states_list, dim=-1)
        return self.projection(concatenated)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def save_adapter(adapter: nn.Module, path: str | Path):
    """Save adapter weights as safetensors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(adapter.state_dict(), str(path))
    print(f"Saved adapter ({adapter.total_params:,} params) to {path}")


def load_adapter(
    path: str | Path,
    adapter_type: str = "per_layer",
    source_dim: int = 1536,
    target_dim: int = 2560,
    num_layers: int = 3,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load adapter weights from safetensors."""
    path = Path(path)
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
    print(f"Loaded adapter ({adapter.total_params:,} params) from {path}")
    return adapter


# Quick sanity check
if __name__ == "__main__":
    print("=== Adapter Module Sanity Check ===")

    batch, seq_len = 2, 77

    # Per-layer adapter
    adapter_pl = PerLayerProjection(1536, 2560, 3)
    dummy_hidden = [torch.randn(batch, seq_len, 1536) for _ in range(3)]
    out = adapter_pl(dummy_hidden)
    print(f"PerLayerProjection: {[h.shape for h in dummy_hidden]} → {out.shape}")
    print(f"  Parameters: {adapter_pl.total_params:,}")
    assert out.shape == (batch, seq_len, 7680)

    # Concat adapter
    adapter_cc = ConcatProjection(1536, 7680, 3)
    out2 = adapter_cc(dummy_hidden)
    print(f"ConcatProjection:   {[h.shape for h in dummy_hidden]} → {out2.shape}")
    print(f"  Parameters: {adapter_cc.total_params:,}")
    assert out2.shape == (batch, seq_len, 7680)

    print("✓ All shapes correct")
