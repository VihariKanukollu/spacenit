"""Panopticon model adapter for SpaceNit benchmarks."""

from dataclasses import dataclass

from torch import nn


@dataclass
class PanopticonConfig:
    """Configuration for the Panopticon model adapter."""

    # TODO: Populate with Panopticon-specific configuration fields
    pass


class Panopticon(nn.Module):
    """Panopticon model adapter for benchmark evaluation.

    TODO: Implement the full Panopticon model adapter.
    """

    def __init__(self, config: PanopticonConfig | None = None) -> None:
        """Initialize Panopticon adapter."""
        super().__init__()
        self.config = config or PanopticonConfig()

    def forward(self, *args, **kwargs):
        """Forward pass."""
        raise NotImplementedError("Panopticon adapter not yet implemented")

    def forward_features(self, *args, **kwargs):
        """Forward features pass for spatial pooling."""
        raise NotImplementedError("Panopticon adapter not yet implemented")
