"""Feature extraction from models."""

import logging

import torch
from torch.utils.data import DataLoader

from spacenit.benchmarks.benchmark_adapter import BenchmarkAdapter
from spacenit.benchmarks.feature_transforms import quantize_features
from spacenit.structures import MaskedGeoSample

logger = logging.getLogger(__name__)


def get_features(
    data_loader: DataLoader,
    model: BenchmarkAdapter,
    is_train: bool = True,
    quantize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get features from model for the data in data_loader.

    Args:
        data_loader: DataLoader for the evaluation dataset.
        model: BenchmarkAdapter-wrapped model to get features from.
        is_train: Whether this is training data (affects some model behaviors).
        quantize: If True, quantize features to int8 for storage efficiency testing.

    Returns:
        Tuple of (features, labels). If quantize=True, features are int8.
    """
    features_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    model.eval()
    device = model.device
    total_samples = len(data_loader)
    with torch.no_grad():
        for i, (sample, label) in enumerate(data_loader):
            sample_dict = sample.as_dict(ignore_nones=True)
            for key, val in sample_dict.items():
                if key == "timestamps":
                    sample_dict[key] = val.to(device=device)
                else:
                    sample_dict[key] = val.to(device=device)

            sample = MaskedGeoSample.from_dict(sample_dict)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                batch_features, label = model(
                    sample=sample,
                    labels=label,
                    is_train=is_train,
                )

            features_list.append(batch_features.cpu())
            labels_list.append(label)
            logger.info(f"Processed {i} / {total_samples}")

    features = torch.cat(features_list, dim=0)  # (N, dim)
    labels = torch.cat(labels_list, dim=0)  # (N)

    if quantize:
        logger.info(f"Quantizing features from {features.dtype} to int8")
        features = quantize_features(features)

    return features, labels
