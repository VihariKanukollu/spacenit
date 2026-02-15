"""Dual-branch latent predictor architecture.

Two independent decoder branches (alpha and beta) each predict masked latent
representations from different views of the same sample.  A momentum-updated
target encoder provides regression targets.  An optional pixel decoder can
reconstruct raw inputs for auxiliary auto-encoding losses.
"""

import logging
from copy import deepcopy
from dataclasses import dataclass

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


class DualBranchPredictor(nn.Module, ParallelMixin):
    """Dual-branch latent predictor.

    The online encoder produces latent embeddings for visible tokens.  Two
    independent decoder branches (``branch_alpha`` and ``branch_beta``) each
    predict masked latent targets from different masked views.  A
    momentum-updated copy of the encoder (``momentum_encoder``) provides
    regression targets.
    """

    supports_multiple_modalities_at_once = True

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        pixel_decoder: nn.Module | None = None,
    ):
        """Initialise the dual-branch predictor.

        Args:
            encoder: Online encoder that processes visible tokens.
            decoder: Predictor head — will be deep-copied for the second branch.
            pixel_decoder: Optional decoder for raw-pixel reconstruction.
        """
        super().__init__()
        self.encoder = encoder
        self.branch_alpha = decoder
        self.branch_beta = deepcopy(decoder)
        self.momentum_encoder = deepcopy(self.encoder)
        self.pixel_decoder = pixel_decoder
        for p in self.momentum_encoder.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Per-branch forward helpers
    # ------------------------------------------------------------------

    def run_alpha(
        self, x: MaskedGeoSample, patch_size: int
    ) -> tuple[EmbeddingsAndMasks, EmbeddingsAndMasks, torch.Tensor, EmbeddingsAndMasks | None]:
        """Forward pass through the encoder and branch-alpha decoder.

        Args:
            x: A masked geospatial sample.
            patch_size: Spatial patch size used during tokenisation.

        Returns:
            latent: Embeddings produced by the online encoder.
            decoded: Predictions from branch alpha for masked tokens.
            latent_projected_and_pooled: Pooled embeddings for contrastive loss.
            reconstructed: Pixel-decoder predictions (``None`` when disabled).
        """
        output_dict = self.encoder(x, patch_size=patch_size)
        latent, latent_projected_and_pooled, decoder_kwargs = (
            extract_encoder_outputs(output_dict)
        )

        reconstructed = None
        if self.pixel_decoder:
            reconstructed = self.pixel_decoder(latent, x.timestamps, patch_size)

        decoded = self.branch_alpha(
            latent,
            timestamps=x.timestamps,
            patch_size=patch_size,
            **decoder_kwargs,
        )
        return latent, decoded, latent_projected_and_pooled, reconstructed

    def run_beta(
        self, x: MaskedGeoSample, patch_size: int
    ) -> tuple[EmbeddingsAndMasks, EmbeddingsAndMasks, torch.Tensor, EmbeddingsAndMasks | None]:
        """Forward pass through the encoder and branch-beta decoder.

        Args:
            x: A masked geospatial sample.
            patch_size: Spatial patch size used during tokenisation.

        Returns:
            latent: Embeddings produced by the online encoder.
            decoded: Predictions from branch beta for masked tokens.
            latent_projected_and_pooled: Pooled embeddings for contrastive loss.
            reconstructed: Pixel-decoder predictions (``None`` when disabled).
        """
        output_dict = self.encoder(x, patch_size=patch_size)
        latent, latent_projected_and_pooled, decoder_kwargs = (
            extract_encoder_outputs(output_dict)
        )

        reconstructed = None
        if self.pixel_decoder:
            reconstructed = self.pixel_decoder(latent, x.timestamps, patch_size)

        decoded = self.branch_beta(
            latent,
            timestamps=x.timestamps,
            patch_size=patch_size,
            **decoder_kwargs,
        )
        return latent, decoded, latent_projected_and_pooled, reconstructed

    # ------------------------------------------------------------------
    # Combined forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_alpha: MaskedGeoSample,
        input_beta: MaskedGeoSample,
        patch_size: int,
    ) -> dict[
        str,
        tuple[EmbeddingsAndMasks, EmbeddingsAndMasks, torch.Tensor, EmbeddingsAndMasks | None],
    ]:
        """Run both branches and return their outputs keyed by branch name.

        Args:
            input_alpha: Masked sample for the alpha branch.
            input_beta: Masked sample for the beta branch.
            patch_size: Spatial patch size used during tokenisation.

        Returns:
            Dictionary with keys ``"alpha"`` and ``"beta"``, each mapping to
            the corresponding branch's output tuple.
        """
        return {
            "alpha": self.run_alpha(input_alpha, patch_size),
            "beta": self.run_beta(input_beta, patch_size),
        }

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
        self.branch_alpha.apply_fsdp(**fsdp_config)
        self.branch_beta.apply_fsdp(**fsdp_config)
        self.momentum_encoder.apply_fsdp(**fsdp_config)
        if self.pixel_decoder:
            self.pixel_decoder.apply_fsdp(**fsdp_config)
        fully_shard(self, **fsdp_config)
        register_fsdp_forward_method(self.momentum_encoder, "forward")

    def enable_compile(self) -> None:
        """Apply ``torch.compile`` to every sub-module for graph optimisation."""
        self.encoder.apply_compile()
        self.branch_alpha.apply_compile()
        self.branch_beta.apply_compile()
        self.momentum_encoder.apply_compile()
        if self.pixel_decoder is not None:
            self.pixel_decoder.apply_compile()


@dataclass
class DualBranchPredictorConfig(Config):
    """Configuration for :class:`DualBranchPredictor`."""

    encoder_cfg: Config
    decoder_cfg: Config
    pixel_decoder_cfg: Config | None = None

    def validate(self) -> None:
        """Check cross-field consistency."""
        if (
            self.encoder_cfg.supported_modalities
            != self.decoder_cfg.supported_modalities
        ):
            raise ValueError("Encoder and decoder must support the same modalities")
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

    def build(self) -> "DualBranchPredictor":
        """Construct a :class:`DualBranchPredictor` from this config."""
        self.validate()
        encoder = self.encoder_cfg.build()
        decoder = self.decoder_cfg.build()
        pixel_decoder = (
            self.pixel_decoder_cfg.build()
            if self.pixel_decoder_cfg is not None
            else None
        )
        return DualBranchPredictor(
            encoder=encoder,
            decoder=decoder,
            pixel_decoder=pixel_decoder,
        )
