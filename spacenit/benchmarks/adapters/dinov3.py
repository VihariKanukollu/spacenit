"""DINOv3 model adapter for SpaceNit benchmarks."""

from dataclasses import dataclass

from torch import nn


@dataclass
class DINOv3Config:
    """Configuration for the DINOv3 model adapter."""

    # TODO: Populate with DINOv3-specific configuration fields
    pass


class DINOv3(nn.Module):
    """DINOv3 model adapter for benchmark evaluation.

    TODO: Implement the full DINOv3 model adapter.
    """

    def __init__(self, config: DINOv3Config | None = None) -> None:
        """Initialize DINOv3 adapter."""
        super().__init__()
        self.config = config or DINOv3Config()

    def forward(self, *args, **kwargs):
        """Forward pass."""
        raise NotImplementedError("DINOv3 adapter not yet implemented")

    def forward_features(self, *args, **kwargs):
        """Forward features pass for spatial pooling."""
        raise NotImplementedError("DINOv3 adapter not yet implemented")
