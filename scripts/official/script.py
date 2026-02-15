"""SpaceNit training entry point -- shared configuration builders.

Provides the common ``build_*`` functions that per-size scripts
(``nano.py``, ``base.py``, etc.) import and compose.
"""

import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig

from spacenit.arch.common import PoolingType
from spacenit.ingestion.sensors import (
    CDL,
    LANDSAT,
    OPENSTREETMAP_RASTER,
    SENTINEL1,
    SENTINEL2_L2A,
    SRTM,
    WORLDCOVER,
    WORLDCEREAL,
    WRI_CANOPY_HEIGHT_MAP,
)
from spacenit.ingestion.tile_loader import GeoTileLoaderConfig
from spacenit.ingestion.tile_dataset import GeoTileDatasetConfig
from spacenit.ops.launch_config import (
    build_common_components as build_common_components_default,
)
from spacenit.ops.experiment import (
    CommonComponents,
    SpaceNitVisualizeConfig,
    SubCmd,
)
from spacenit.pipeline.hooks.downstream_eval import (
    DownstreamEvalHookConfig,
    DownstreamTaskConfig,
)
from spacenit.pipeline.hooks.throughput_monitor import SpaceNitThroughputMonitor
from spacenit.pipeline.hooks.experiment_logger import SpaceNitExperimentLogger
from spacenit.pipeline.runners.contrastive_latent import (
    ContrastiveLatentRunnerConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build the common components for an experiment."""
    config = build_common_components_default(script, cmd, run_name, cluster, overrides)
    config.training_modalities = [
        SENTINEL2_L2A.label,
        SENTINEL1.label,
        LANDSAT.label,
        WORLDCOVER.label,
        SRTM.label,
        OPENSTREETMAP_RASTER.label,
        WRI_CANOPY_HEIGHT_MAP.label,
        CDL.label,
        WORLDCEREAL.label,
    ]
    return config


def get_masking_config() -> dict:
    """Build the masking strategy configuration.

    Uses cross-sensor random masking: 50 % of tokens are visible to the
    encoder, 50 % are prediction targets.  Low-information sensors
    (land-cover, elevation, etc.) are decode-only.
    """
    return {
        "type": "cross_sensor_random",
        "encode_ratio": 0.5,
        "decode_ratio": 0.5,
        "decode_only_sensors": [
            WORLDCOVER.label,
            SRTM.label,
            OPENSTREETMAP_RASTER.label,
            WRI_CANOPY_HEIGHT_MAP.label,
            CDL.label,
            WORLDCEREAL.label,
        ],
    }


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentRunnerConfig:
    """Build the train module config for an experiment.

    Args:
        common: Common experiment components.
    """
    return ContrastiveLatentRunnerConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=32,
        masking_config=get_masking_config(),
        contrastive_temperature=0.1,
        uniformity_weight=0.1,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=8000),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_dataloader_config(
    common: CommonComponents,
) -> GeoTileLoaderConfig:
    """Build the dataloader config for an experiment.

    Args:
        common: Common experiment components.
    """
    return GeoTileLoaderConfig(
        num_workers=12,
        global_batch_size=512,
        token_budget=2250,
        prefetch_factor=2,
        sampled_hw_p_list=list(range(1, 13)),
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
        num_masked_views=2,
        tokenization_config=common.tokenization_config,
    )


def build_dataset_config(common: CommonComponents) -> GeoTileDatasetConfig:
    """Build the dataset config for an experiment."""
    return GeoTileDatasetConfig(
        h5py_dir="/path/to/your/h5py_dataset",
        training_modalities=common.training_modalities,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(300)
    METRICS_COLLECT_INTERVAL = 10
    CANCEL_CHECK_INTERVAL = 25
    LOAD_STRATEGY = LoadStrategy.if_available
    WANDB_USERNAME = "spacenit"  # nosec
    WANDB_PROJECT = "2025_10_02_phase2"
    PERMANENT_SAVE_INTERVAL = 5000
    EPHERMERAL_SAVE_INTERVAL = 250
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = SpaceNitExperimentLogger(
        name=common.run_name,
        project=WANDB_PROJECT,
        entity=WANDB_USERNAME,
        enabled=True,
    )
    garbage_collector_callback = GarbageCollectorCallback(gc_interval=1)
    EVAL_TASKS = {
        "m-eurosat": DownstreamTaskConfig(
            dataset="m-eurosat",
            embedding_batch_size=128,
            num_workers=0,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.steps(4000),
        ),
        "m_so2sat": DownstreamTaskConfig(
            dataset="m-so2sat",
            embedding_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.steps(20000),
        ),
        "mados": DownstreamTaskConfig(
            dataset="mados",
            embedding_batch_size=128,
            probe_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=False,
            probe_lr=0.01,
            epochs=50,
            eval_interval=Duration.steps(4000),
        ),
        "pastis": DownstreamTaskConfig(
            dataset="pastis",
            embedding_batch_size=32,
            probe_batch_size=8,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.steps(20000),
            input_modalities=[SENTINEL2_L2A.label],
            epochs=50,
        ),
    }
    trainer_config = (
        TrainerConfig(
            work_dir=common.save_folder,
            load_strategy=LOAD_STRATEGY,
            save_folder=common.save_folder,
            cancel_check_interval=CANCEL_CHECK_INTERVAL,
            metrics_collect_interval=METRICS_COLLECT_INTERVAL,
            max_duration=MAX_DURATION,
            checkpointer=checkpointer_config,
        )
        .with_callback("wandb", wandb_callback)
        .with_callback("speed_monitor", SpaceNitThroughputMonitor())
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvalHookConfig(
                tasks=EVAL_TASKS,
            ),
        )
        .with_callback("garbage_collector", garbage_collector_callback)
        .with_callback(
            "beaker", BeakerCallback()
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=PERMANENT_SAVE_INTERVAL,
                ephemeral_save_interval=EPHERMERAL_SAVE_INTERVAL,
            ),
        )
    )
    return trainer_config


def build_visualize_config(common: CommonComponents) -> SpaceNitVisualizeConfig:
    """Build the visualize config for an experiment."""
    return SpaceNitVisualizeConfig(
        num_samples=None,
        output_dir=str(f"{common.save_folder}/visualizations"),
        std_multiplier=2.0,
    )
