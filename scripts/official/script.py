"""SpaceNit training entry point — shared configuration builders."""

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
from spacenit.arch.adaptive_vision_encoder import (
    PoolingType,
)
from spacenit.pipeline.hooks.downstream_eval import (
    DownstreamEvalHookConfig,
    DownstreamTaskConfig,
)
from spacenit.pipeline.hooks.throughput_monitor import SpaceNitThroughputMonitor
from spacenit.pipeline.hooks.experiment_logger import SpaceNitExperimentLogger
from spacenit.pipeline.objectives import ObjectiveConfig
from spacenit.pipeline.occlusion import OcclusionConfig
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


def get_occlusion_config(common: CommonComponents) -> OcclusionConfig:
    """Get the occlusion configuration for the experiment.

    Args:
        common: Common experiment components containing optional tokenization_config.
    """
    return OcclusionConfig(
        strategy_config={
            "type": "modality_cross_random",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "allow_encoding_decoding_same_bandset": True,
            "only_decode_modalities": [
                WORLDCOVER.label,
                SRTM.label,
                OPENSTREETMAP_RASTER.label,
                WRI_CANOPY_HEIGHT_MAP.label,
                CDL.label,
                WORLDCEREAL.label,
            ],
        },
        tokenization_config=common.tokenization_config,
    )


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentRunnerConfig:
    """Build the train module config for an experiment.

    Args:
        common: Common experiment components.
    """
    # The train module still needs the occlusion_config for reference (e.g., for metric
    # naming), but the actual masking happens in the dataloader workers.
    return ContrastiveLatentRunnerConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=32,
        occlusion_config=get_occlusion_config(common),
        loss_config=ObjectiveConfig(
            loss_config={
                "type": "modality_patch_discrimination_new",
                "tau": 0.1,
            }
        ),
        contrastive_config=ObjectiveConfig(
            loss_config={
                "type": "InfoNCE",
                "weight": 0.1,
            }
        ),
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=8000),
        ema_decay=(1.0, 1.0),
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

    Masking is performed in the dataloader workers (CPU) instead of in the train module
    (GPU). This improves throughput by offloading CPU-bound masking operations to
    dataloader workers.

    Args:
        common: Common experiment components.
    """
    return GeoTileLoaderConfig(
        num_workers=12,
        global_batch_size=512,
        token_budget=2250,
        prefetch_factor=2,
        sampled_hw_p_list=list(range(1, 13)),  # try only temporal tokens
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
        num_masked_views=2,  # ContrastiveLatentRunner needs 2 views
        occlusion_config=get_occlusion_config(common),
        # occlusion_config_b is not set, so both views use the same strategy
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
        enabled=True,  # set to False to avoid wandb errors
    )
    # Safe to collect every step for now
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
