"""SpaceNit benchmark datasets.

Dataset classes are imported lazily inside :func:`get_benchmark_dataset`
to avoid pulling in heavy optional dependencies (rioxarray, etc.) at
package-import time.
"""

from __future__ import annotations

import logging

from olmo_core.config import StrEnum
from torch.utils.data import Dataset

from .band_scaling import NormMethod

logger = logging.getLogger(__name__)


class BenchmarkDatasetPartition(StrEnum):
    """Enum for different dataset partitions."""

    TRAIN1X = "default"
    TRAIN_001X = "0.01x_train"
    TRAIN_002X = "0.02x_train"
    TRAIN_005X = "0.05x_train"
    TRAIN_010X = "0.10x_train"
    TRAIN_020X = "0.20x_train"
    TRAIN_050X = "0.50x_train"


def get_benchmark_dataset(
    benchmark_dataset: str,
    split: str,
    norm_stats_from_pretrained: bool = False,
    input_modalities: list[str] | None = None,
    input_layers: list[str] | None = None,
    partition: str = BenchmarkDatasetPartition.TRAIN1X,
    norm_method: str = NormMethod.NORM_NO_CLIP,
) -> Dataset:
    """Retrieve a benchmark dataset from the dataset name.

    Dataset classes are imported lazily to avoid pulling in heavy
    optional dependencies at package-import time.
    """
    import spacenit.benchmarks.datasets.storage as storage

    if input_modalities is None:
        input_modalities = []
    if input_layers is None:
        input_layers = []

    if input_modalities:
        if benchmark_dataset not in ["pastis", "pastis128", "nandi", "awf"]:
            raise ValueError(
                f"input_modalities is only supported for multimodal tasks, got {benchmark_dataset}"
            )

    if input_layers:
        if benchmark_dataset not in ["nandi", "awf"]:
            raise ValueError(
                f"input_layers is only supported for rslearn tasks, got {benchmark_dataset}"
            )

    if benchmark_dataset.startswith("m-"):
        from .geo_benchmark import GeoBenchmarkDataset

        return GeoBenchmarkDataset(
            geobench_dir=storage.GEOBENCH_DIR,
            dataset=benchmark_dataset,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif benchmark_dataset == "mados":
        from .marine_debris import MarineDebrisDataset

        if norm_stats_from_pretrained:
            logger.warning(
                "MADOS has very different norm stats than our pretraining dataset"
            )
        return MarineDebrisDataset(
            path_to_splits=storage.MADOS_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif benchmark_dataset == "sen1floods11":
        from .flood_scenes import FloodScenesDataset

        return FloodScenesDataset(
            path_to_splits=storage.FLOODS_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif benchmark_dataset.startswith("pastis"):
        from .crop_parcels import CropParcelsDataset

        kwargs = {
            "split": split,
            "partition": partition,
            "norm_stats_from_pretrained": norm_stats_from_pretrained,
            "input_modalities": input_modalities,
            "norm_method": norm_method,
            "dir_partition": storage.PASTIS_DIR_PARTITION,
        }
        if "128" in benchmark_dataset:
            kwargs["path_to_splits"] = storage.PASTIS_DIR_ORIG
        else:
            kwargs["path_to_splits"] = storage.PASTIS_DIR
        return CropParcelsDataset(**kwargs)  # type: ignore
    elif benchmark_dataset == "breizhcrops":
        from .crop_timeseries import CropTimeseriesDataset

        return CropTimeseriesDataset(
            path_to_splits=storage.BREIZHCROPS_DIR,
            split=split,
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
        )
    elif benchmark_dataset == "nandi":
        from .rslearn_adapter import RslearnToSpaceNitDataset

        return RslearnToSpaceNitDataset(
            ds_path=storage.NANDI_DIR,
            ds_groups=["groundtruth_polygon_split_window_32"],
            layers=input_layers,
            input_size=4,
            split=split,
            property_name="category",
            classes=["Coffee", "Trees", "Grassland", "Maize", "Sugarcane", "Tea"],
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
            input_modalities=input_modalities,
            start_time="2022-09-01",
            end_time="2023-09-01",
            ds_norm_stats_json="nandi_band_stats.json",
        )
    elif benchmark_dataset == "awf":
        from .rslearn_adapter import RslearnToSpaceNitDataset

        return RslearnToSpaceNitDataset(
            ds_path=storage.AWF_DIR,
            ds_groups=["20250822"],
            layers=input_layers,
            input_size=32,
            split=split,
            property_name="lulc",
            classes=[
                "Agriculture/Settlement",
                "Grassland/barren",
                "Herbaceous wetland",
                "Lava forest",
                "Montane forest",
                "Open water",
                "Shrubland/Savanna",
                "Urban/dense development",
                "Woodland forest (>40% canopy)",
            ],
            partition=partition,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            norm_method=norm_method,
            input_modalities=input_modalities,
            start_time="2023-01-01",
            end_time="2023-12-31",
            ds_norm_stats_json="awf_band_stats.json",
        )
    else:
        raise ValueError(f"Unrecognized benchmark_dataset {benchmark_dataset}")
