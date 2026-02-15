"""High-level model compositions for different pretraining objectives.

Each model is a thin composition of encoder, decoder, and head modules:

- :class:`LatentPredictor` -- encoder + momentum encoder + decoder (JEPA-style)
- :class:`AutoEncoder` -- encoder + pixel reconstruction head (MAE-style)
- :class:`DualBranch` -- encoder + two decoders for dual-objective training
- :class:`SpatioTemporalEncoder` -- encoder with axial attention factorization

Design choices vs. the original codebase:
- EMA (momentum) update logic lives in the model, not the runner
- Axial attention (attend along H, then W, then T separately) instead of
  alternating spatial/temporal blocks
- Models are simple compositions, not deep inheritance hierarchies
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor

from spacenit.arch.attention import RMSNorm, TransformerBlock, TransformerStack
from spacenit.arch.encoder import Decoder, Encoder, EncoderConfig
from spacenit.arch.helpers import ParallelMixin
from spacenit.arch.heads import PixelHead, PoolingHead, ProjectionHead
from spacenit.settings import Config


# ---------------------------------------------------------------------------
# Exponential Moving Average helper
# ---------------------------------------------------------------------------


@torch.no_grad()
def _ema_update(
    online: nn.Module,
    target: nn.Module,
    momentum: float,
) -> None:
    """Update *target* parameters as an exponential moving average of *online*.

    ``target_param = momentum * target_param + (1 - momentum) * online_param``
    """
    # Under FSDP2 composable sharding, params may be DTensors. EMA must update
    # the target params using the *same* DTensor layout as the online params.
    target_by_name = dict(target.named_parameters())
    for name, p_online in online.named_parameters():
        p_target = target_by_name.get(name)
        if p_target is None:
            continue

        online_data = p_online.data
        target_data = p_target.data

        if isinstance(online_data, DTensor) or isinstance(target_data, DTensor):
            if not isinstance(online_data, DTensor) or not isinstance(target_data, DTensor):
                raise RuntimeError(
                    f"EMA expected both online and target params to be DTensors for '{name}', "
                    f"got online={type(online_data)} target={type(target_data)}"
                )

            # Align target layout to online layout (this fixes Shard vs Replicate mismatches).
            if target_data.device_mesh != online_data.device_mesh:
                target_data = target_data.redistribute(
                    device_mesh=online_data.device_mesh, placements=online_data.placements
                )
            elif target_data.placements != online_data.placements:
                target_data = target_data.redistribute(placements=online_data.placements)

            p_target.data = target_data
            target_data.mul_(momentum).add_(online_data, alpha=1.0 - momentum)
            continue

        # Non-DTensor path.
        target_data.mul_(momentum).add_(online_data, alpha=1.0 - momentum)


# ---------------------------------------------------------------------------
# Latent Predictor (JEPA-style)
# ---------------------------------------------------------------------------


@dataclass
class LatentPredictorConfig(Config):
    """Configuration for :class:`LatentPredictor`."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder_depth: int = 4
    decoder_num_heads: int = 12
    decoder_num_kv_heads: int | None = None
    ema_momentum: float = 0.996
    ema_momentum_end: float = 1.0
    ema_warmup_steps: int = 10000
    projection_dim: int = 256

    def build(self) -> LatentPredictor:
        """Build a :class:`LatentPredictor` from this configuration."""
        return LatentPredictor(self)


class LatentPredictor(ParallelMixin, nn.Module):
    """Encoder + momentum encoder + decoder for latent prediction.

    The online encoder processes visible tokens.  The momentum (target)
    encoder processes all tokens to produce prediction targets.  The
    decoder predicts the target encoder's representations for masked
    positions.

    EMA update logic is embedded in the model itself (via :meth:`step_ema`)
    rather than being duplicated across training runners.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: LatentPredictorConfig) -> None:
        super().__init__()
        self.config = config

        # Online encoder
        self.encoder = Encoder(config.encoder)

        # Momentum (target) encoder -- deep copy, no gradients
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Decoder
        self.decoder = Decoder(
            embed_dim=config.encoder.embed_dim,
            depth=config.decoder_depth,
            num_heads=config.decoder_num_heads,
            num_kv_heads=config.decoder_num_kv_heads,
        )

        # Projection heads (online and target)
        self.online_proj = ProjectionHead(
            in_dim=config.encoder.embed_dim,
            out_dim=config.projection_dim,
        )
        self.target_proj = ProjectionHead(
            in_dim=config.encoder.embed_dim,
            out_dim=config.projection_dim,
        )
        # Target projection also uses EMA
        for p in self.target_proj.parameters():
            p.requires_grad = False

        self._ema_momentum = config.ema_momentum
        self.register_buffer(
            "_step", torch.tensor(0, dtype=torch.long), persistent=True
        )

    @torch.no_grad()
    def step_ema(self) -> None:
        """Perform one EMA update step.

        Momentum is linearly annealed from ``ema_momentum`` to
        ``ema_momentum_end`` over ``ema_warmup_steps``.
        """
        cfg = self.config
        step_val = self._step.item()
        progress = min(step_val / max(cfg.ema_warmup_steps, 1), 1.0)
        momentum = cfg.ema_momentum + (cfg.ema_momentum_end - cfg.ema_momentum) * progress
        _ema_update(self.encoder, self.target_encoder, momentum)
        _ema_update(self.online_proj, self.target_proj, momentum)
        self._ema_momentum = momentum
        self._step.add_(1)

    # ------------------------------------------------------------------
    # Distributed / compile helpers (expected by BaseRunner)
    # ------------------------------------------------------------------

    def apply_ddp(
        self,
        *,
        dp_mesh: Any | None = None,
        compile_enabled: bool = False,
        find_unused_parameters: bool = True,
    ) -> None:
        self.enable_ddp(
            dp_mesh=dp_mesh,
            compile_enabled=compile_enabled,
            find_unused_parameters=find_unused_parameters,
        )

    def apply_fsdp(
        self,
        *,
        dp_mesh: Any | None = None,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype = torch.float32,
    ) -> None:
        # FSDP2 composable API.
        #
        # Important: downstream eval sometimes calls `model.encoder(...)` directly.
        # If we only shard the top-level module, calling a submodule directly can
        # bypass FSDP unshard/reshard hooks and surface DTensor parameters inside
        # ops (mixed Tensor/DTensor errors). Shard the relevant submodules so
        # they remain FSDP-wrapped even when called directly.
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

        mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
        # Ensure the momentum (target) modules get sharded the same way as the
        # online modules. Some FSDP2 paths may treat `requires_grad=False` params
        # differently (replicated), which breaks EMA updates.
        for p in self.target_encoder.parameters():
            p.requires_grad = True
        for p in self.target_proj.parameters():
            p.requires_grad = True
        fully_shard(self.encoder, mesh=dp_mesh, mp_policy=mp)
        fully_shard(self.target_encoder, mesh=dp_mesh, mp_policy=mp)
        fully_shard(self.decoder, mesh=dp_mesh, mp_policy=mp)
        fully_shard(self.online_proj, mesh=dp_mesh, mp_policy=mp)
        fully_shard(self.target_proj, mesh=dp_mesh, mp_policy=mp)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_proj.parameters():
            p.requires_grad = False

    def apply_compile(self) -> None:
        # Optional; used only when runner requests compile.
        self.encoder = torch.compile(self.encoder)  # type: ignore[assignment]
        self.decoder = torch.compile(self.decoder)  # type: ignore[assignment]
        self.online_proj = torch.compile(self.online_proj)  # type: ignore[assignment]

    def forward(
        self,
        sensor_data: dict[str, Tensor],
        *,
        visible_indices: Tensor | None = None,
        target_indices: Tensor | None = None,
        patch_size: int | None = None,
        month_indices: Tensor | None = None,
        gsd: float = 10.0,
        contrastive_only: bool = False,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            sensor_data: Mapping from sensor label to data tensor.
            visible_indices: ``(B, N_vis)`` indices of visible tokens for
                the online encoder.
            target_indices: ``(B, N_tgt)`` indices of tokens to predict.
            patch_size: Runtime patch size.
            month_indices: Month-of-year indices.
            gsd: Ground-sample distance.
            contrastive_only: If ``True``, skip the decoder and produce
                mean-pooled encoder representations for contrastive
                learning.  Returns ``"online_proj"`` and
                ``"target_proj"`` only (both ``(B, projection_dim)``).

        Returns:
            Dictionary with keys (full mode):
            - ``"predictions"`` -- decoder output ``(B, N_tgt, D)``.
            - ``"targets"`` -- target encoder output ``(B, N_tgt, D)``.
            - ``"online_proj"`` -- projected online representations.
            - ``"target_proj"`` -- projected target representations.

            Or (contrastive_only mode):
            - ``"online_proj"`` -- ``(B, projection_dim)`` L2-normalised.
            - ``"target_proj"`` -- ``(B, projection_dim)`` L2-normalised.
        """
        # Online encoder (visible tokens only)
        encoded, sensor_ids, layout = self.encoder(
            sensor_data,
            patch_size=patch_size,
            month_indices=month_indices,
            gsd=gsd,
            mask_indices=visible_indices,
        )

        # Target encoder (all tokens, no gradient)
        with torch.no_grad():
            target_encoded, _, _ = self.target_encoder(
                sensor_data,
                patch_size=patch_size,
                month_indices=month_indices,
                gsd=gsd,
            )

        # ----------------------------------------------------------
        # Contrastive-only mode: pool + project, skip decoder
        # ----------------------------------------------------------
        if contrastive_only:
            online_pooled = encoded.mean(dim=1)       # (B, D)
            online_z = self.online_proj(online_pooled)

            with torch.no_grad():
                target_pooled = target_encoded.mean(dim=1)
                target_z = self.target_proj(target_pooled)

            return {
                "online_proj": online_z,
                "target_proj": target_z.detach(),
            }

        # ----------------------------------------------------------
        # Full mode: decode + project
        # ----------------------------------------------------------

        # Extract target representations at prediction positions
        num_predictions = target_indices.shape[1] if target_indices is not None else 0
        if target_indices is not None and num_predictions > 0:
            B, _, D = target_encoded.shape
            idx = target_indices.unsqueeze(-1).expand(-1, -1, D)
            targets = torch.gather(target_encoded, dim=1, index=idx)
        else:
            targets = target_encoded

        # Decode
        predictions = self.decoder(
            encoded, num_predictions=num_predictions
        )

        # Project
        online_proj = self.online_proj(predictions)
        with torch.no_grad():
            target_proj = self.target_proj(targets)

        return {
            "predictions": predictions,
            "targets": targets.detach(),
            "online_proj": online_proj,
            "target_proj": target_proj.detach(),
        }


# ---------------------------------------------------------------------------
# Auto-Encoder (MAE-style)
# ---------------------------------------------------------------------------


@dataclass
class AutoEncoderConfig(Config):
    """Configuration for :class:`AutoEncoder`."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder_depth: int = 4
    decoder_num_heads: int = 12
    decoder_num_kv_heads: int | None = None
    out_channels: int = 13  # e.g. Sentinel-2 bands

    def build(self) -> AutoEncoder:
        """Build an :class:`AutoEncoder` from this configuration."""
        return AutoEncoder(self)


class AutoEncoder(nn.Module):
    """Encoder + decoder + pixel reconstruction head (MAE-style).

    Args:
        config: Model configuration.
    """

    def __init__(self, config: AutoEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(
            embed_dim=config.encoder.embed_dim,
            depth=config.decoder_depth,
            num_heads=config.decoder_num_heads,
            num_kv_heads=config.decoder_num_kv_heads,
        )
        self.pixel_head = PixelHead(
            embed_dim=config.encoder.embed_dim,
            patch_size=config.encoder.base_patch_size,
            out_channels=config.out_channels,
        )

    def forward(
        self,
        sensor_data: dict[str, Tensor],
        *,
        visible_indices: Tensor | None = None,
        masked_indices: Tensor | None = None,
        patch_size: int | None = None,
        month_indices: Tensor | None = None,
        gsd: float = 10.0,
        height: int = 224,
        width: int = 224,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Returns:
            Dictionary with ``"reconstructed"`` pixel output and
            ``"encoded"`` latent representations.
        """
        encoded, sensor_ids, layout = self.encoder(
            sensor_data,
            patch_size=patch_size,
            month_indices=month_indices,
            gsd=gsd,
            mask_indices=visible_indices,
        )

        num_masked = masked_indices.shape[1] if masked_indices is not None else 0
        decoded = self.decoder(encoded, num_predictions=num_masked)

        reconstructed = self.pixel_head(decoded, height, width)

        return {
            "reconstructed": reconstructed,
            "encoded": encoded,
        }


# ---------------------------------------------------------------------------
# Dual-Branch Predictor
# ---------------------------------------------------------------------------


@dataclass
class DualBranchConfig(Config):
    """Configuration for :class:`DualBranch`."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder_depth: int = 4
    decoder_num_heads: int = 12
    decoder_num_kv_heads: int | None = None
    ema_momentum: float = 0.996
    ema_momentum_end: float = 1.0
    ema_warmup_steps: int = 10000
    projection_dim: int = 256
    out_channels: int = 13

    def build(self) -> DualBranch:
        """Build a :class:`DualBranch` from this configuration."""
        return DualBranch(self)


class DualBranch(ParallelMixin, nn.Module):
    """Encoder + two decoders for dual-objective training.

    Combines latent prediction (JEPA-style) with pixel reconstruction
    (MAE-style) using shared encoder weights.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: DualBranchConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.encoder.embed_dim

        # Shared encoder
        self.encoder = Encoder(config.encoder)

        # Momentum encoder
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Latent prediction decoder
        self.latent_decoder = Decoder(
            embed_dim=dim,
            depth=config.decoder_depth,
            num_heads=config.decoder_num_heads,
            num_kv_heads=config.decoder_num_kv_heads,
        )

        # Pixel reconstruction decoder
        self.pixel_decoder = Decoder(
            embed_dim=dim,
            depth=config.decoder_depth,
            num_heads=config.decoder_num_heads,
            num_kv_heads=config.decoder_num_kv_heads,
        )
        self.pixel_head = PixelHead(
            embed_dim=dim,
            patch_size=config.encoder.base_patch_size,
            out_channels=config.out_channels,
        )

        # Projection heads
        self.online_proj = ProjectionHead(in_dim=dim, out_dim=config.projection_dim)
        self.target_proj = ProjectionHead(in_dim=dim, out_dim=config.projection_dim)
        for p in self.target_proj.parameters():
            p.requires_grad = False

        self.register_buffer(
            "_step", torch.tensor(0, dtype=torch.long), persistent=True
        )

    @torch.no_grad()
    def step_ema(self) -> None:
        """Perform one EMA update step."""
        cfg = self.config
        step_val = self._step.item()
        progress = min(step_val / max(cfg.ema_warmup_steps, 1), 1.0)
        momentum = cfg.ema_momentum + (cfg.ema_momentum_end - cfg.ema_momentum) * progress
        _ema_update(self.encoder, self.target_encoder, momentum)
        _ema_update(self.online_proj, self.target_proj, momentum)
        self._step.add_(1)

    def apply_ddp(
        self,
        *,
        dp_mesh: Any | None = None,
        compile_enabled: bool = False,
        find_unused_parameters: bool = True,
    ) -> None:
        self.enable_ddp(
            dp_mesh=dp_mesh,
            compile_enabled=compile_enabled,
            find_unused_parameters=find_unused_parameters,
        )

    def apply_fsdp(
        self,
        *,
        dp_mesh: Any | None = None,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype = torch.float32,
    ) -> None:
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

        mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
        fully_shard(self, mesh=dp_mesh, mp_policy=mp)

    def apply_compile(self) -> None:
        self.encoder = torch.compile(self.encoder)  # type: ignore[assignment]
        self.latent_decoder = torch.compile(self.latent_decoder)  # type: ignore[assignment]
        self.pixel_decoder = torch.compile(self.pixel_decoder)  # type: ignore[assignment]
        self.online_proj = torch.compile(self.online_proj)  # type: ignore[assignment]

    def forward(
        self,
        sensor_data: dict[str, Tensor],
        *,
        visible_indices: Tensor | None = None,
        target_indices: Tensor | None = None,
        patch_size: int | None = None,
        month_indices: Tensor | None = None,
        gsd: float = 10.0,
        height: int = 224,
        width: int = 224,
    ) -> dict[str, Tensor]:
        """Forward pass producing both latent predictions and pixel reconstructions."""
        # Online encoder
        encoded, sensor_ids, layout = self.encoder(
            sensor_data,
            patch_size=patch_size,
            month_indices=month_indices,
            gsd=gsd,
            mask_indices=visible_indices,
        )

        # Target encoder
        with torch.no_grad():
            target_encoded, _, _ = self.target_encoder(
                sensor_data,
                patch_size=patch_size,
                month_indices=month_indices,
                gsd=gsd,
            )

        num_targets = target_indices.shape[1] if target_indices is not None else 0

        # Extract targets
        if target_indices is not None and num_targets > 0:
            B, _, D = target_encoded.shape
            idx = target_indices.unsqueeze(-1).expand(-1, -1, D)
            targets = torch.gather(target_encoded, dim=1, index=idx)
        else:
            targets = target_encoded

        # Latent prediction branch
        latent_pred = self.latent_decoder(encoded, num_predictions=num_targets)
        online_proj = self.online_proj(latent_pred)
        with torch.no_grad():
            target_proj = self.target_proj(targets)

        # Pixel reconstruction branch
        pixel_pred = self.pixel_decoder(encoded, num_predictions=num_targets)
        reconstructed = self.pixel_head(pixel_pred, height, width)

        return {
            "latent_predictions": latent_pred,
            "targets": targets.detach(),
            "online_proj": online_proj,
            "target_proj": target_proj.detach(),
            "reconstructed": reconstructed,
            "encoded": encoded,
        }


# ---------------------------------------------------------------------------
# Spatio-Temporal Encoder (Axial Attention)
# ---------------------------------------------------------------------------


class AxialAttentionBlock(nn.Module):
    """Factorized attention: attend along H, then W, then T separately.

    This is a genuinely different algorithm from the original approach of
    alternating spatial/temporal transformer blocks.  Axial attention
    reduces the quadratic cost of full 3D attention to linear in each
    dimension.

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads for GQA.
        ffn_expansion: SwiGLU expansion ratio.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
        ffn_expansion: float = 8 / 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Three separate transformer blocks for each axis
        self.h_block = TransformerBlock(
            dim, num_heads, num_kv_heads, ffn_expansion, dropout
        )
        self.w_block = TransformerBlock(
            dim, num_heads, num_kv_heads, ffn_expansion, dropout
        )
        self.t_block = TransformerBlock(
            dim, num_heads, num_kv_heads, ffn_expansion, dropout
        )

    def forward(
        self,
        x: Tensor,
        H: int,
        W: int,
        T: int,
    ) -> Tensor:
        """Apply axial attention.

        Args:
            x: ``(B, H*W*T, D)`` flattened spatio-temporal tokens.
            H: Number of spatial rows (patches).
            W: Number of spatial columns (patches).
            T: Number of temporal steps.

        Returns:
            ``(B, H*W*T, D)`` with axial attention applied.
        """
        B, N, D = x.shape
        assert N == H * W * T, f"Expected {H*W*T} tokens, got {N}"

        # Reshape to (B, H, W, T, D)
        x = x.view(B, H, W, T, D)

        # Attend along H: merge W,T into batch -> (B*W*T, H, D)
        x = x.permute(0, 2, 3, 1, 4).reshape(B * W * T, H, D)
        x = self.h_block(x)
        x = x.view(B, W, T, H, D).permute(0, 3, 1, 2, 4)  # (B, H, W, T, D)

        # Attend along W: merge H,T into batch -> (B*H*T, W, D)
        x = x.permute(0, 1, 3, 2, 4).reshape(B * H * T, W, D)
        x = self.w_block(x)
        x = x.view(B, H, T, W, D).permute(0, 1, 3, 2, 4)  # (B, H, W, T, D)

        # Attend along T: merge H,W into batch -> (B*H*W, T, D)
        x = x.reshape(B * H * W, T, D)
        x = self.t_block(x)
        x = x.view(B, H, W, T, D)

        # Flatten back to (B, H*W*T, D)
        return x.reshape(B, N, D)


@dataclass
class SpatioTemporalConfig(Config):
    """Configuration for :class:`SpatioTemporalEncoder`."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    axial_depth: int = 6
    projection_dim: int = 256

    def build(self) -> SpatioTemporalEncoder:
        """Build a :class:`SpatioTemporalEncoder` from this configuration."""
        return SpatioTemporalEncoder(self)


class SpatioTemporalEncoder(ParallelMixin, nn.Module):
    """Encoder with factorized axial attention for spatio-temporal data.

    Uses :class:`AxialAttentionBlock` to attend along spatial height,
    spatial width, and temporal dimensions separately.  This reduces the
    O(N^2) cost of full attention over H*W*T tokens to O(H^2 + W^2 + T^2).

    Args:
        config: Model configuration.
    """

    def __init__(self, config: SpatioTemporalConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.encoder.embed_dim

        # Standard encoder for initial tokenization
        self.encoder = Encoder(config.encoder)

        # Axial attention layers
        self.axial_layers = nn.ModuleList([
            AxialAttentionBlock(
                dim=dim,
                num_heads=config.encoder.num_heads,
                num_kv_heads=config.encoder.num_kv_heads,
                ffn_expansion=config.encoder.ffn_expansion,
                dropout=config.encoder.dropout,
            )
            for _ in range(config.axial_depth)
        ])

        self.norm = RMSNorm(dim)

        # Pooling and projection
        self.pool = PoolingHead(
            embed_dim=dim,
            num_heads=config.encoder.num_heads,
            num_kv_heads=config.encoder.num_kv_heads,
        )
        self.proj = ProjectionHead(
            in_dim=dim, out_dim=config.projection_dim
        )

    def apply_ddp(
        self,
        *,
        dp_mesh: Any | None = None,
        compile_enabled: bool = False,
        find_unused_parameters: bool = True,
    ) -> None:
        self.enable_ddp(
            dp_mesh=dp_mesh,
            compile_enabled=compile_enabled,
            find_unused_parameters=find_unused_parameters,
        )

    def apply_fsdp(
        self,
        *,
        dp_mesh: Any | None = None,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype = torch.float32,
    ) -> None:
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

        mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
        fully_shard(self, mesh=dp_mesh, mp_policy=mp)

    def apply_compile(self) -> None:
        self.encoder = torch.compile(self.encoder)  # type: ignore[assignment]

    def forward(
        self,
        sensor_data: dict[str, Tensor],
        *,
        H: int,
        W: int,
        T: int,
        patch_size: int | None = None,
        month_indices: Tensor | None = None,
        gsd: float = 10.0,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            sensor_data: Sensor data tensors.
            H: Number of spatial patch rows.
            W: Number of spatial patch columns.
            T: Number of temporal steps.
            patch_size: Runtime patch size.
            month_indices: Month-of-year indices.
            gsd: Ground-sample distance.

        Returns:
            Dictionary with ``"encoded"``, ``"pooled"``, and ``"projected"``
            representations.
        """
        # Initial encoding (standard transformer)
        encoded, sensor_ids, layout = self.encoder(
            sensor_data,
            patch_size=patch_size,
            month_indices=month_indices,
            gsd=gsd,
        )

        # Separate sensor-type tokens (spatial_id == -1) from spatial tokens.
        # The tokenizer prepends one sensor-type token per sensor, so we use
        # the layout to identify them: each sensor contributes 1 type token
        # followed by its spatial tokens.
        type_token_indices: list[int] = []
        spatial_token_indices: list[int] = []
        offset = 0
        for _label, n_tokens in layout:
            type_token_indices.append(offset)  # first token is the type token
            spatial_token_indices.extend(range(offset + 1, offset + n_tokens))
            offset += n_tokens

        B, N, D = encoded.shape
        type_idx = torch.tensor(type_token_indices, device=encoded.device)
        spatial_idx = torch.tensor(spatial_token_indices, device=encoded.device)

        type_tokens = encoded[:, type_idx]        # (B, num_sensors, D)
        spatial_tokens = encoded[:, spatial_idx]   # (B, H*W*T, D)

        # Apply axial attention only to spatial tokens
        for axial_layer in self.axial_layers:
            spatial_tokens = axial_layer(spatial_tokens, H, W, T)

        # Recombine: restore original token order
        recombined = encoded.clone()
        recombined[:, type_idx] = type_tokens
        recombined[:, spatial_idx] = spatial_tokens

        recombined = self.norm(recombined)

        # Pool and project (over all tokens including type tokens)
        pooled = self.pool(recombined)
        projected = self.proj(pooled)

        return {
            "encoded": recombined,
            "pooled": pooled,
            "projected": projected,
        }
