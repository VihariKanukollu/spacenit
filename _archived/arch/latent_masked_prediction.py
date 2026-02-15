"""Latent masked prediction architecture.

Predicts masked token representations in a learned latent space using a
momentum-updated target encoder and a lightweight predictor head.  An optional
pixel decoder can reconstruct raw inputs for auxiliary auto-encoding losses.
"""

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)

from spacenit.settings import Config
from spacenit.structures import MaskedGeoSample
from spacenit.arch.adaptive_vision_encoder import EmbeddingsAndMasks
from spacenit.arch.helpers import extract_encoder_outputs, ParallelMixin

logger = logging.getLogger(__name__)


class LatentMaskedPredictor(nn.Module, ParallelMixin):
    """Latent masked-prediction model.

    The online encoder produces latent embeddings for visible tokens.  A
    momentum-updated copy of the encoder (``momentum_encoder``) provides
    regression targets for the masked positions.  A lightweight decoder
    predicts the latent targets, and an optional ``pixel_decoder`` can
    reconstruct the raw pixel values.
    """

    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        pixel_decoder: nn.Module | None = None,
    ):
        """Initialise the latent masked predictor.

        Args:
            encoder: Online encoder that processes visible tokens.
            decoder: Lightweight predictor that regresses masked latents.
            pixel_decoder: Optional decoder for raw-pixel reconstruction.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pixel_decoder = pixel_decoder
        self.momentum_encoder = deepcopy(self.encoder)
        for p in self.momentum_encoder.parameters():
            p.requires_grad = False

    def forward(
        self, x: MaskedGeoSample, patch_size: int
    ) -> tuple[
        EmbeddingsAndMasks,
        EmbeddingsAndMasks,
        torch.Tensor,
        EmbeddingsAndMasks | None,
        dict[str, Any],
    ]:
        """Run one forward pass of the latent masked predictor.

        Args:
            x: A masked geospatial sample.
            patch_size: Spatial patch size used during tokenisation.

        Returns:
            latent: Embeddings produced by the online encoder.
            decoded: Predictions from the decoder for masked tokens.
            latent_projected_and_pooled: Pooled embeddings for contrastive loss.
            reconstructed: Pixel-decoder predictions (``None`` when disabled).
            extra_metrics: Auxiliary metrics such as token-norm statistics.
        """
        output_dict = self.encoder(x, patch_size=patch_size)
        token_norm_stats = output_dict.pop("token_norm_stats", None)
        latent, latent_projected_and_pooled, decoder_kwargs = (
            extract_encoder_outputs(output_dict)
        )

        extra_metrics: dict[str, Any] = {}
        if token_norm_stats is not None:
            extra_metrics["token_norm_stats"] = token_norm_stats

        reconstructed = None
        if self.pixel_decoder:
            reconstructed = self.pixel_decoder(latent, x.timestamps, patch_size)

        decoded = self.decoder(
            latent,
            timestamps=x.timestamps,
            patch_size=patch_size,
            **decoder_kwargs,
        )

        return (
            latent,
            decoded,
            latent_projected_and_pooled,
            reconstructed,
            extra_metrics,
        )

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
        self.decoder.apply_fsdp(**fsdp_config)
        self.momentum_encoder.apply_fsdp(**fsdp_config)
        if self.pixel_decoder:
            self.pixel_decoder.apply_fsdp(**fsdp_config)
        fully_shard(self, **fsdp_config)
        register_fsdp_forward_method(self.momentum_encoder, "forward")

    def enable_compile(self) -> None:
        """Apply ``torch.compile`` to every sub-module for graph optimisation."""
        logger.info("Applying torch.compile to the model")
        self.encoder.apply_compile()
        logger.info("Applied torch.compile to the encoder")
        self.decoder.apply_compile()
        logger.info("Applied torch.compile to the decoder")
        self.momentum_encoder.apply_compile()
        logger.info("Applied torch.compile to the momentum encoder")


@dataclass
class LatentMaskedPredictorConfig(Config):
    """Configuration for :class:`LatentMaskedPredictor`."""

    encoder_cfg: Config
    decoder_cfg: Config
    pixel_decoder_cfg: Config | None = None

    def validate(self) -> None:
        """Check cross-field consistency."""
        if (
            self.encoder_cfg.supported_sensor_labels
            != self.decoder_cfg.supported_sensor_labels
        ):
            raise ValueError("Encoder and decoder must support the same sensors")
        if (
            self.encoder_cfg.max_sequence_length
            != self.decoder_cfg.max_sequence_length
        ):
            raise ValueError(
                "Encoder and decoder must have the same max sequence length"
            )
        if (
            self.encoder_cfg.embed_dim
            != self.decoder_cfg.encoder_embed_dim
        ):
            raise ValueError("Encoder embedding size must be consistent!")

    def build(self) -> "LatentMaskedPredictor":
        """Construct a :class:`LatentMaskedPredictor` from this config."""
        self.validate()
        encoder = self.encoder_cfg.build()
        decoder = self.decoder_cfg.build()
        pixel_decoder = (
            self.pixel_decoder_cfg.build()
            if self.pixel_decoder_cfg is not None
            else None
        )
        return LatentMaskedPredictor(
            encoder=encoder,
            decoder=decoder,
            pixel_decoder=pixel_decoder,
        )
