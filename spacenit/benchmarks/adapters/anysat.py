"""AnySat model adapter for SpaceNit benchmarks."""

from dataclasses import dataclass

from torch import nn


@dataclass
class AnySatConfig:
    """Configuration for the AnySat model adapter."""

    # TODO: Populate with AnySat-specific configuration fields
    pass


class AnySat(nn.Module):
    """AnySat model adapter for benchmark evaluation.

    TODO: Implement the full AnySat model adapter.
    """

    def __init__(self, config: AnySatConfig | None = None) -> None:
        """Initialize AnySat adapter."""
        super().__init__()
        self.config = config or AnySatConfig()

    def forward(self, *args, **kwargs):
        """Forward pass."""
        raise NotImplementedError("AnySat adapter not yet implemented")
