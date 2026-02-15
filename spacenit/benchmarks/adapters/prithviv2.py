"""PrithviV2 model adapter for SpaceNit benchmarks."""

from dataclasses import dataclass

from torch import nn


@dataclass
class PrithviV2Config:
    """Configuration for the PrithviV2 model adapter."""

    # TODO: Populate with PrithviV2-specific configuration fields
    pass


class PrithviV2(nn.Module):
    """PrithviV2 model adapter for benchmark evaluation.

    TODO: Implement the full PrithviV2 model adapter.
    """

    def __init__(self, config: PrithviV2Config | None = None) -> None:
        """Initialize PrithviV2 adapter."""
        super().__init__()
        self.config = config or PrithviV2Config()

    def forward(self, *args, **kwargs):
        """Forward pass."""
        raise NotImplementedError("PrithviV2 adapter not yet implemented")
