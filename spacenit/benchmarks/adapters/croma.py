"""Croma model adapter for SpaceNit benchmarks."""

from dataclasses import dataclass

from torch import nn


@dataclass
class CromaConfig:
    """Configuration for the Croma model adapter."""

    # TODO: Populate with Croma-specific configuration fields
    pass


class Croma(nn.Module):
    """Croma model adapter for benchmark evaluation.

    TODO: Implement the full Croma model adapter.
    """

    def __init__(self, config: CromaConfig | None = None) -> None:
        """Initialize Croma adapter."""
        super().__init__()
        self.config = config or CromaConfig()

    def forward(self, *args, **kwargs):
        """Forward pass."""
        raise NotImplementedError("Croma adapter not yet implemented")
