"""Masked auto-encoder architecture.

Implements a standard masked auto-encoder (MAE) that reconstructs missing
patches from visible ones.  Supports an optional lightweight latent decoder
alongside the primary pixel decoder.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)

from spacenit.settings import Config
from spacenit.structures import MaskedGeoSample
from spacenit.arch.adaptive_vision_encoder import EmbeddingsAndMasks
from spacenit.arch.helpers import extract_encoder_outputs, ParallelMixin


class MaskedAutoEncoder(nn.Module, ParallelMixin):
    """Masked Auto-Encoder module.

    The encoder processes only the visible tokens.  A decoder predicts the
    latent representations for masked positions, and an optional pixel decoder
    reconstructs the raw input values.
    """

    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module | None = None,
        pixel_decoder: nn.Module | None = None,
    ):
        """Initialise the masked auto-encoder.

        Args:
            encoder: Vision encoder that processes visible tokens.
            decoder: Optional latent predictor for masked positions.
            pixel_decoder: Optional decoder that reconstructs raw pixel values.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pixel_decoder = pixel_decoder

    def forward(
        self, x: MaskedGeoSample, patch_size: int
    ) -> tuple[EmbeddingsAndMasks, EmbeddingsAndMasks | None, EmbeddingsAndMasks | None]:
        """Run one forward pass of the masked auto-encoder.

        Args:
            x: A masked geospatial sample.
            patch_size: Spatial patch size used during tokenisation.

        Returns:
            latent: Embeddings produced by the encoder.
            decoded: Latent predictions for masked tokens (``None`` if no decoder).
            reconstructed: Pixel-level reconstruction (``None`` if no pixel decoder).
        """
        output_dict = self.encoder(x, patch_size=patch_size)
        latent, _, decoder_kwargs = extract_encoder_outputs(output_dict)

        decoded = self.decoder and self.decoder(
            latent,
            timestamps=x.timestamps,
            patch_size=patch_size,
            **decoder_kwargs,
        )
        reconstructed = self.pixel_decoder and self.pixel_decoder(
            latent, timestamps=x.timestamps, patch_size=patch_size
        )

        return latent, decoded, reconstructed

    # ------------------------------------------------------------------
    # Distributed-training helpers
    # ------------------------------------------------------------------

    def enable_fsdp(
        self,
        dp_mesh: DeviceMesh | None = None,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype = torch.float32,
        prefetch_factor: int = 0,
    ) -> None:
        """Wrap every sub-module with Fully Sharded Data Parallel."""
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)

        self.encoder.apply_fsdp(**fsdp_config)
        if self.decoder:
            self.decoder.apply_fsdp(**fsdp_config)
        if self.pixel_decoder:
            self.pixel_decoder.apply_fsdp(**fsdp_config)
        fully_shard(self, **fsdp_config)

    def enable_compile(self) -> None:
        """Apply ``torch.compile`` to every sub-module for graph optimisation."""
        self.encoder.apply_compile()
        if self.decoder is not None:
            self.decoder.apply_compile()
        if self.pixel_decoder is not None:
            self.pixel_decoder.apply_compile()


@dataclass
class MaskedAutoEncoderConfig(Config):
    """Configuration for :class:`MaskedAutoEncoder`."""

    encoder_cfg: Config
    decoder_cfg: Config | None = None
    pixel_decoder_cfg: Config | None = None

    def validate(self) -> None:
        """Check cross-field consistency."""
        if self.decoder_cfg is not None:
            if (
                self.encoder_cfg.supported_modalities
                != self.decoder_cfg.supported_modalities
            ):
                raise ValueError(
                    "Encoder and decoder must support the same modalities"
                )
            if (
                self.encoder_cfg.max_sequence_length
                != self.decoder_cfg.max_sequence_length
            ):
                raise ValueError(
                    "Encoder and decoder must have the same max sequence length"
                )
            if (
                self.encoder_cfg.embedding_size
                != self.decoder_cfg.encoder_embedding_size
            ):
                raise ValueError("Encoder embedding size must be consistent!")
        if self.pixel_decoder_cfg is not None:
            if (
                self.encoder_cfg.supported_modalities
                != self.pixel_decoder_cfg.supported_modalities
            ):
                raise ValueError(
                    "Encoder and pixel decoder must support the same modalities"
                )
            if (
                self.encoder_cfg.max_sequence_length
                != self.pixel_decoder_cfg.decoder_config.max_sequence_length
            ):
                raise ValueError(
                    "Encoder and pixel decoder must have the same max sequence length"
                )
            if (
                self.encoder_cfg.embedding_size
                != self.pixel_decoder_cfg.decoder_config.encoder_embedding_size
            ):
                raise ValueError("Encoder embedding size must be consistent!")

    def build(self) -> "MaskedAutoEncoder":
        """Construct a :class:`MaskedAutoEncoder` from this config."""
        self.validate()
        encoder = self.encoder_cfg.build()
        decoder = (
            self.decoder_cfg.build() if self.decoder_cfg is not None else None
        )
        pixel_decoder = (
            self.pixel_decoder_cfg.build()
            if self.pixel_decoder_cfg is not None
            else None
        )
        return MaskedAutoEncoder(
            encoder=encoder,
            decoder=decoder,
            pixel_decoder=pixel_decoder,
        )
