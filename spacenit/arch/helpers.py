"""Shared utilities for the architecture module.

Small helper functions and mix-in classes used across the neural-network
components in :mod:`spacenit.arch`.
"""

from typing import Any

import torch
from torch.distributed import DeviceMesh


def extract_encoder_outputs(
    output_dict: dict[str, Any],
) -> tuple:
    """Decompose an encoder's output dictionary into its canonical parts.

    Pops the well-known keys (latent tokens, projected/pooled
    representation, normalisation statistics) and returns everything else
    as pass-through keyword arguments for a downstream decoder.

    Args:
        output_dict: Dictionary produced by an encoder's forward pass.
            Modified in-place – consumed keys are removed.

    Returns:
        A three-element tuple of:

        * **latent** – the raw token-and-mask container (or ``None``).
        * **latent_projected_and_pooled** – the projected and aggregated
          representation (or ``None``).
        * **decoder_kwargs** – remaining entries forwarded to the decoder.
    """
    latent = output_dict.pop("tokens_and_masks", None)
    latent_projected_and_pooled = output_dict.pop("project_aggregated", None)
    # Normalisation statistics are consumed but not forwarded.
    output_dict.pop("token_norm_stats", None)
    decoder_kwargs = output_dict
    return latent, latent_projected_and_pooled, decoder_kwargs


def cumulative_seq_offsets(seq_lengths: torch.Tensor) -> torch.Tensor:
    """Convert per-sequence lengths into cumulative start offsets.

    Zero-length entries are silently skipped so that the result is
    compatible with flash-attention's variable-length kernels.

    Args:
        seq_lengths: 1-D integer tensor of individual sequence lengths.

    Returns:
        1-D ``int32`` tensor of length ``(K + 1)`` where *K* is the number
        of non-zero entries in *seq_lengths*.  The first element is always
        ``0`` and the last equals the total number of tokens.
    """
    return torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=seq_lengths.device),
            torch.cumsum(
                seq_lengths.masked_select(seq_lengths != 0), 0, dtype=torch.int32
            ),
        ]
    )


class ParallelMixin:
    """Mix-in that adds distributed-training helpers to a model.

    Intended to be combined with :class:`torch.nn.Module` via multiple
    inheritance.  Provides a single method for activating DDP replication.
    """

    def enable_ddp(
        self,
        dp_mesh: DeviceMesh | None = None,
        compile_enabled: bool = False,
        autograd_compile_enabled: bool = False,
        find_unused_parameters: bool = True,
    ) -> None:
        """Wrap the model with composable Distributed Data Parallelism.

        Uses :func:`torch.distributed._composable.replicate.replicate`
        under the hood, which is compatible with ``torch.compile`` and
        FSDP2.

        Args:
            dp_mesh: Device mesh describing the data-parallel group.
                ``None`` uses the default process group.
            compile_enabled: When ``True``, configures ``torch._dynamo``
                for optimal DDP + compile interaction.
            autograd_compile_enabled: When ``True`` *and*
                *compile_enabled* is also ``True``, selects the
                ``python_reducer_without_compiled_forward`` DDP optimiser
                strategy.
            find_unused_parameters: Pass ``True`` when not every parameter
                receives a gradient in every step (e.g. masked
                auto-encoders).

        .. note::
            This method is normally called by the top-level model builder
            and does not need to be invoked directly.
        """
        from torch.distributed._composable.replicate import replicate

        if compile_enabled:
            if autograd_compile_enabled:
                torch._dynamo.config.optimize_ddp = (
                    "python_reducer_without_compiled_forward"  # type: ignore
                )
            else:
                torch._dynamo.config.optimize_ddp = "ddp_optimizer"  # type: ignore

        replicate(
            self,
            device_mesh=dp_mesh,
            bucket_cap_mb=100,
            find_unused_parameters=find_unused_parameters,
        )
