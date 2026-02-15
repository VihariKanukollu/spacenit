"""Adaptive patch embedding and reconstruction modules.

Provides :class:`AdaptivePatchEmbedding` for converting 2-D (or 2-D + time)
image tensors into patch-level embeddings at arbitrary patch sizes, and
:class:`AdaptivePatchReconstruction` for the inverse operation.  Both modules
support *flexible* patch sizes at inference time by resizing either the input
image or the reconstructed patches so that the learned convolutional kernel can
be reused across resolutions.

Extended from `pytorch-image-models`_ patch-embed layer with ideas from
FlexiViT_.

.. _pytorch-image-models:
   https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py
.. _FlexiViT:
   https://github.com/bwconrad/flexivit/
"""

import logging
from collections.abc import Iterable
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from spacenit.ingestion.sensors import SensorSpec

logger = logging.getLogger(__name__)


class AdaptivePatchEmbedding(nn.Module):
    """Convert a 2-D image (with optional time axis) into patch embeddings.

    The module learns a single convolutional kernel whose spatial extent equals
    ``base_patch_size * sensor_spec.tile_size_multiplier``.  At inference time a
    different ``patch_size`` can be requested; the input is resized so that the
    fixed kernel produces the correct number of output patches.

    Attributes:
        embed_dim: Dimensionality of each output patch embedding.
        sensor_spec: Sensor specification describing the input modality.
        patch_size: Effective kernel / stride size after accounting for the
            sensor's ``tile_size_multiplier``.
        conv_proj: Learned 2-D convolution that projects each patch to
            ``embed_dim`` dimensions.
        norm: Optional normalisation applied after projection.
        resize_mode: Interpolation algorithm used when adapting to a
            non-native patch size (e.g. ``"bicubic"``).
        smooth_resize: Whether anti-aliasing is applied during interpolation.
    """

    def __init__(
        self,
        sensor_spec: SensorSpec,
        base_patch_size: int | tuple[int, int],
        input_channels: int = 3,
        embed_dim: int = 128,
        norm_layer: nn.Module | None = None,
        bias: bool = True,
        resize_mode: str = "bicubic",
        smooth_resize: bool = True,
    ) -> None:
        """Initialise the adaptive patch embedding layer.

        Args:
            sensor_spec: :class:`SensorSpec` instance that describes the input
                sensor.  Its ``tile_size_multiplier`` is used to scale
                ``base_patch_size`` to the true pixel-space patch extent.
            base_patch_size: Patch size expressed at the *base* tile resolution
                (i.e. before applying ``tile_size_multiplier``).  A single
                integer is broadcast to a square ``(size, size)`` pair.
            input_channels: Number of channels in the input image.
            embed_dim: Dimensionality of the output embedding for each patch.
            norm_layer: Optional normalisation constructor (e.g.
                ``nn.LayerNorm``).  When *None*, an identity layer is used.
            bias: If ``True``, the projection convolution includes a bias term.
            resize_mode: Interpolation mode passed to
                :func:`torch.nn.functional.interpolate` when the requested
                patch size differs from the native one.
            smooth_resize: Whether to enable anti-aliased interpolation.
        """
        super().__init__()

        self.embed_dim = embed_dim

        self.sensor_spec = sensor_spec
        self.patch_size = self._ensure_pair(
            base_patch_size * sensor_spec.tile_size_multiplier
        )

        self.conv_proj = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Adaptive-resolution attributes
        self.resize_mode = resize_mode
        self.smooth_resize = smooth_resize

    @staticmethod
    def _ensure_pair(x: Any) -> tuple[int, int]:
        """Normalise *x* to a 2-tuple ``(height, width)``.

        If *x* is already an iterable of length 2 it is returned as a tuple;
        a scalar value is duplicated to form a square pair.

        Args:
            x: Value to convert.  May be a single integer for square patches or
                an iterable of exactly two integers for rectangular patches.

        Returns:
            A ``(height, width)`` tuple.

        Raises:
            AssertionError: If *x* is an iterable whose length is not 2.
        """
        if isinstance(x, Iterable) and not isinstance(x, str):
            assert len(list(x)) == 2, "x must be a 2-tuple"
            return tuple(x)
        return (x, x)

    def forward(
        self,
        x: Tensor,
        patch_size: int | tuple[int, int] | None = None,
    ) -> Tensor:
        """Project an image tensor into patch embeddings.

        The input is expected in *channels-last* layout and may optionally
        include a temporal axis:

        * 4-D input: ``[batch, height, width, channels]``
        * 5-D input: ``[batch, height, width, time, channels]``

        When a ``patch_size`` other than the native one is requested the input
        is spatially resized so that the fixed convolutional kernel produces the
        correct grid of patches.

        Args:
            x: Input image tensor with shape ``[B, H, W, C]`` or
                ``[B, H, W, T, C]``.
            patch_size: Desired patch size at the *base* tile resolution.  If
                ``None``, the native ``self.patch_size`` is used (i.e. the size
                determined at construction time from ``base_patch_size`` and the
                sensor's ``tile_size_multiplier``).

        Returns:
            Patch embeddings with shape ``[B, H', W', D]`` (no time axis) or
            ``[B, H', W', T, D]`` (with time axis), where ``H'`` and ``W'``
            are the spatial grid dimensions and ``D`` is ``embed_dim``.
        """
        batch_size = x.shape[0]
        has_time_dimension = False
        num_timesteps = 0  # ignored when has_time_dimension is False

        if len(x.shape) == 5:
            has_time_dimension = True
            num_timesteps = x.shape[3]
            x = rearrange(x, "b h w t c -> (b t) c h w")
        else:
            x = rearrange(x, "b h w c -> b c h w")

        if not patch_size:
            # During evaluation use the native patch size when none is given.
            patch_size = self.patch_size
        else:
            if isinstance(patch_size, tuple):
                patch_size = (
                    patch_size[0] * self.sensor_spec.tile_size_multiplier,
                    patch_size[1] * self.sensor_spec.tile_size_multiplier,
                )
            else:
                patch_size = patch_size * self.sensor_spec.tile_size_multiplier
        patch_size = self._ensure_pair(patch_size)
        assert isinstance(patch_size, tuple) and len(patch_size) == 2, (
            "patch_size must be a 2-tuple"
        )

        # Resize the input when the requested patch size differs from native.
        if patch_size != self.patch_size:
            shape = x.shape[-2:]
            new_shape = (
                shape[0] // patch_size[0] * self.patch_size[0],
                shape[1] // patch_size[1] * self.patch_size[1],
            )
            x = F.interpolate(
                x,
                size=new_shape,
                mode=self.resize_mode,
                antialias=self.smooth_resize,
            )

        # Project each patch to embed_dim via the learned convolution.
        x = self.conv_proj(x)

        # Rearrange back to channels-last layout.
        if has_time_dimension:
            _, d, h, w = x.shape
            x = rearrange(
                x,
                "(b t) d h w -> b h w t d",
                b=batch_size,
                t=num_timesteps,
                d=d,
                h=h,
                w=w,
            )
        else:
            x = rearrange(x, "b d h w -> b h w d")

        x = self.norm(x)

        return x


class AdaptivePatchReconstruction(nn.Module):
    """Reconstruct a 2-D image from patch embeddings with flexible patch sizes.

    This is the inverse of :class:`AdaptivePatchEmbedding`.  A transposed
    convolution maps each ``embed_dim``-dimensional patch vector back to a
    pixel block of size ``largest_patch_size``.  When the desired output patch
    size is smaller than the kernel, the reconstructed patches are spatially
    interpolated to the target resolution.

    Attributes:
        embed_dim: Dimensionality of each input patch embedding.
        largest_patch_size: Maximum (native) patch size supported by the
            transposed convolution kernel.
        deconv_proj: Learned transposed convolution that maps embeddings back
            to pixel space.
        norm: Optional normalisation applied after reconstruction.
        resize_mode: Interpolation algorithm used when the target patch size
            is smaller than ``largest_patch_size``.
        smooth_resize: Whether anti-aliasing is applied during interpolation.
    """

    def __init__(
        self,
        largest_patch_size: int | tuple[int, int],
        output_channels: int = 3,
        embed_dim: int = 128,
        norm_layer: nn.Module | None = None,
        bias: bool = True,
        resize_mode: str = "bicubic",
        smooth_resize: bool = True,
    ) -> None:
        """Initialise the adaptive patch reconstruction layer.

        Args:
            largest_patch_size: The maximum patch size (in pixels) that the
                transposed convolution kernel is designed for.  A single integer
                is broadcast to a square ``(size, size)`` pair.
            output_channels: Number of channels in the reconstructed image.
            embed_dim: Dimensionality of the input patch embeddings.
            norm_layer: Optional normalisation constructor (e.g.
                ``nn.LayerNorm``).  When *None*, an identity layer is used.
            bias: If ``True``, the transposed convolution includes a bias term.
            resize_mode: Interpolation mode passed to
                :func:`torch.nn.functional.interpolate` when the requested
                patch size is smaller than ``largest_patch_size``.
            smooth_resize: Whether to enable anti-aliased interpolation.
        """
        super().__init__()

        self.embed_dim = embed_dim

        self.largest_patch_size = self._ensure_pair(largest_patch_size)

        self.deconv_proj = nn.ConvTranspose2d(
            embed_dim,
            output_channels,
            kernel_size=largest_patch_size,
            stride=largest_patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Adaptive-resolution attributes
        self.resize_mode = resize_mode
        self.smooth_resize = smooth_resize

    @staticmethod
    def _ensure_pair(x: Any) -> tuple[int, int]:
        """Normalise *x* to a 2-tuple ``(height, width)``.

        If *x* is already an iterable of length 2 it is returned as a tuple;
        a scalar value is duplicated to form a square pair.

        Args:
            x: Value to convert.  May be a single integer for square patches or
                an iterable of exactly two integers for rectangular patches.

        Returns:
            A ``(height, width)`` tuple.

        Raises:
            AssertionError: If *x* is an iterable whose length is not 2.
        """
        if isinstance(x, Iterable) and not isinstance(x, str):
            assert len(list(x)) == 2, "x must be a 2-tuple"
            return tuple(x)
        return (x, x)

    def _interpolate_kernel(self, x: Tensor, shape: tuple[int, int]) -> Tensor:
        """Resize a single kernel or patch tensor to the target spatial shape.

        A pair of leading singleton dimensions is temporarily added so that
        :func:`~torch.nn.functional.interpolate` can operate on the 2-D
        spatial extent.

        Args:
            x: Input tensor of shape ``[H, W]`` (or broadcastable).
            shape: Target ``(height, width)`` to resize to.

        Returns:
            Resized tensor with the same number of dimensions as *x*.
        """
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.resize_mode,
            antialias=self.smooth_resize,
        )
        return x_resized[0, 0, ...]

    def forward(
        self,
        x: Tensor,
        patch_size: int | tuple[int, int] | None = None,
    ) -> Tensor:
        """Reconstruct an image from patch embeddings.

        The input is expected in *channels-last* layout and may optionally
        include a temporal axis:

        * 4-D input: ``[batch, grid_h, grid_w, embed_dim]``
        * 5-D input: ``[batch, grid_h, grid_w, time, embed_dim]``

        When the requested ``patch_size`` is smaller than
        ``largest_patch_size``, each reconstructed patch block is spatially
        interpolated down to the desired resolution.

        Args:
            x: Patch embedding tensor with shape ``[B, H, W, D]`` or
                ``[B, H, W, T, D]``.
            patch_size: Desired output patch size in pixels.  If ``None``, the
                native ``largest_patch_size`` is used.

        Returns:
            Reconstructed image tensor with shape
            ``[B, H_out, W_out, C]`` (no time axis) or
            ``[B, H_out, W_out, T, C]`` (with time axis), where ``H_out`` and
            ``W_out`` are the full pixel-space spatial dimensions.
        """
        if len(x.shape) == 4:
            has_time_dimension = False
            b, h, w, d = x.shape
            t = 1
        else:
            has_time_dimension = True
            b, h, w, t, d = x.shape

        if not patch_size:
            # During evaluation use the native patch size when none is given.
            patch_size = self.largest_patch_size

        patch_size = self._ensure_pair(patch_size)

        if has_time_dimension:
            x = rearrange(x, "b h w t d -> (b t) d h w", b=b, t=t)
        else:
            x = rearrange(x, "b h w d -> b d h w")

        x = self.deconv_proj(x)

        if patch_size != self.largest_patch_size:
            x = rearrange(
                x,
                "b c (h p_h) (w p_w) -> b h w c p_h p_w",
                p_h=self.largest_patch_size[0],
                p_w=self.largest_patch_size[1],
            )
            bl, hl, wl, cl = x.shape[:4]
            x = rearrange(x, "b h w c p_h p_w -> (b h w) c p_h p_w")
            x = F.interpolate(
                x, patch_size, mode=self.resize_mode, antialias=self.smooth_resize
            )
            x = rearrange(
                x, "(b h w) c p_h p_w -> b c (h p_h) (w p_w)", b=bl, h=hl, w=wl
            )

        if has_time_dimension:
            x = rearrange(x, "(b t) c h w -> b h w t c", b=b, t=t)
        else:
            x = rearrange(x, "b c h w -> b h w c")

        x = self.norm(x)

        return x
