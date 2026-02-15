"""Satlas model adapter for SpaceNit benchmarks."""

from dataclasses import dataclass

from torch import nn


@dataclass
class SatlasConfig:
    """Configuration for the Satlas model adapter."""

    # TODO: Populate with Satlas-specific configuration fields
    pass


class Satlas(nn.Module):
    """Satlas model adapter for benchmark evaluation.

    TODO: Implement the full Satlas model adapter.
    """

    def __init__(self, config: SatlasConfig | None = None) -> None:
        """Initialize Satlas adapter."""
        super().__init__()
        self.config = config or SatlasConfig()

    def forward(self, *args, **kwargs):
        """Forward pass."""
        raise NotImplementedError("Satlas adapter not yet implemented")
