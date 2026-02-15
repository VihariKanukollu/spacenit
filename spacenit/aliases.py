"""Shared type aliases used across the SpaceNit package."""

from typing import TypeAlias

import numpy as np
import torch

# A value that can be either a NumPy array or a PyTorch tensor.
NdTensor: TypeAlias = np.ndarray | torch.Tensor
