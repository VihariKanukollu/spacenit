"""A common home for all benchmark dataset configs."""

from enum import Enum
from typing import NamedTuple

from spacenit.data.constants import Sensor


class TaskType(Enum):
    """Possible task types."""

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"


def get_benchmark_mode(task_type: TaskType) -> str:
    """Get the benchmark mode for a given task type."""
    if task_type == TaskType.CLASSIFICATION:
        return "knn"
    else:
        return "linear_probe"


class BenchmarkDatasetConfig(NamedTuple):
    """BenchmarkDatasetConfig configs."""

    task_type: TaskType
    imputes: list[tuple[str, str]]
    num_classes: int
    is_multilabel: bool
    supported_modalities: list[str]
    # this is only necessary for segmentation tasks,
    # and defines the input / output height width.
    height_width: int | None = None
    timeseries: bool = False


DATASET_TO_CONFIG = {
    "m-eurosat": BenchmarkDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=10,
        is_multilabel=False,
        supported_modalities=[Sensor.SENTINEL2_L2A.label],
    ),
    "m-bigearthnet": BenchmarkDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[("11 - SWIR", "10 - SWIR - Cirrus")],
        num_classes=43,
        is_multilabel=True,
        supported_modalities=[Sensor.SENTINEL2_L2A.label],
    ),
    "m-so2sat": BenchmarkDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[
            ("02 - Blue", "01 - Coastal aerosol"),
            ("08A - Vegetation Red Edge", "09 - Water vapour"),
            ("11 - SWIR", "10 - SWIR - Cirrus"),
        ],
        num_classes=17,
        is_multilabel=False,
        supported_modalities=[Sensor.SENTINEL2_L2A.label],
    ),
    "m-brick-kiln": BenchmarkDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        supported_modalities=[Sensor.SENTINEL2_L2A.label],
    ),
    "m-sa-crop-type": BenchmarkDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[("11 - SWIR", "10 - SWIR - Cirrus")],
        num_classes=10,
        is_multilabel=False,
        height_width=256,
        supported_modalities=[Sensor.SENTINEL2_L2A.label],
    ),
    "m-cashew-plant": BenchmarkDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[("11 - SWIR", "10 - SWIR - Cirrus")],
        num_classes=7,
        is_multilabel=False,
        height_width=256,
        supported_modalities=[Sensor.SENTINEL2_L2A.label],
    ),
    "m-forestnet": BenchmarkDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[
            # src (we have), tgt (we want), using the geobench L8 names
            ("02 - Blue", "01 - Coastal aerosol"),
            ("07 - SWIR2", "09 - Cirrus"),
            ("07 - SWIR2", "10 - Tirs1"),
        ],
        num_classes=12,
        is_multilabel=False,
        supported_modalities=[Sensor.LANDSAT.label],
    ),
    "mados": BenchmarkDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[
            ("05 - Vegetation Red Edge", "06 - Vegetation Red Edge"),
            ("08A - Vegetation Red Edge", "09 - Water vapour"),
            ("11 - SWIR", "10 - SWIR - Cirrus"),
        ],
        num_classes=15,
        is_multilabel=False,
        height_width=80,
        supported_modalities=[Sensor.SENTINEL2_L2A.label],
    ),
    "sen1floods11": BenchmarkDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=2,
        is_multilabel=False,
        height_width=64,
        supported_modalities=[Sensor.SENTINEL1.label],
    ),
    "pastis": BenchmarkDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=19,
        is_multilabel=False,
        height_width=64,
        supported_modalities=[Sensor.SENTINEL2_L2A.label, Sensor.SENTINEL1.label],
        timeseries=True,
    ),
    "pastis128": BenchmarkDatasetConfig(
        task_type=TaskType.SEGMENTATION,
        imputes=[],
        num_classes=19,
        is_multilabel=False,
        height_width=128,
        supported_modalities=[Sensor.SENTINEL2_L2A.label, Sensor.SENTINEL1.label],
        timeseries=True,
    ),
    "breizhcrops": BenchmarkDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=9,
        is_multilabel=False,
        height_width=1,
        supported_modalities=[Sensor.SENTINEL2_L2A.label],
        timeseries=True,
    ),
    "nandi": BenchmarkDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=6,
        is_multilabel=False,
        supported_modalities=[
            Sensor.SENTINEL2_L2A.label,
            Sensor.SENTINEL1.label,
            Sensor.LANDSAT.label,
        ],
        timeseries=True,
    ),
    "awf": BenchmarkDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=9,
        is_multilabel=False,
        supported_modalities=[
            Sensor.SENTINEL2_L2A.label,
            Sensor.SENTINEL1.label,
            Sensor.LANDSAT.label,
        ],
        timeseries=True,
    ),
}


def dataset_to_config(dataset: str) -> BenchmarkDatasetConfig:
    """Retrieve the correct config for a given dataset."""
    try:
        return DATASET_TO_CONFIG[dataset]
    except KeyError:
        raise ValueError(f"Unrecognized dataset: {dataset}")
