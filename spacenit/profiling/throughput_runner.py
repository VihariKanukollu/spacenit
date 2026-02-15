"""Script for performing an inference throughput benchmarking run."""

import itertools
import os
import time
import uuid
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from olmo_core.io import copy_file, file_exists, join_path
from olmo_core.train.callbacks import ProfilerCallback, WandBCallback
from olmo_core.train.trainer import PathOrStr

from spacenit.arch.encoder import Encoder, EncoderConfig
from spacenit.arch.models import LatentPredictorConfig
from spacenit.ingestion.sensors import (
    LANDSAT,
    REFERENCE_GROUND_RESOLUTION,
    SENTINEL1,
    SENTINEL2,
    SENTINEL2_L2A,
)
from spacenit.ops.helpers import MODEL_SIZE_ARGS
from spacenit.profiling import constants
from spacenit.profiling.run_config import RunParams
from spacenit.settings import Config
from spacenit.structures import MaskedGeoSample, TokenVisibility

NUM_S1_BANDS = SENTINEL1.total_channels
NUM_S2_BANDS = SENTINEL2.total_channels
NUM_LANDSAT_BANDS = LANDSAT.total_channels

NUM_SQUARE_KM_LAND_IN_WORLD = 149_000_000

logger = getLogger(__name__)


class MinimalTrainer:
    """Minimal trainer that only has the persist_working_file method.

    Allows using callbacks without the full trainer.
    """

    def __init__(
        self, device: torch.device, work_dir: Path, save_folder: Path | None = None
    ):
        """Initializes the minimal trainer."""
        self.device = device
        self.work_dir = work_dir
        if save_folder is None:
            self.save_folder = work_dir
        else:
            self.save_folder = save_folder

    def persist_working_file(self, name: PathOrStr) -> PathOrStr:
        """Persists a working file."""
        if Path(name).is_relative_to(self.work_dir):
            name = Path(name).relative_to(self.work_dir)
        source = join_path(self.work_dir, name)
        target = join_path(self.save_folder, name)
        if source != target:
            copy_file(source, target)
        elif not file_exists(source):
            raise FileNotFoundError(source)
        return target


class SpaceNitEncoder(torch.nn.Module):
    """Thin wrapper around a SpaceNit checkpoint that loads just the encoder."""

    def __init__(self, model_config: Config) -> None:
        """Loads the checkpoint, keeps only the encoder."""
        super().__init__()

        model = model_config.build()
        self.model: Encoder = getattr(model, "encoder")
        self.model.eval()

    def forward(
        self,
        sensor_data: dict[str, torch.Tensor],
        patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, int]]]:
        """Encode sensor data and return (encoded, sensor_ids, layout)."""
        return self.model.forward(sensor_data, patch_size=patch_size)


def build_default_model_config(
    run_params: RunParams, training_modalities: list[str]
) -> LatentPredictorConfig:
    """Default model config builder based on model_size.

    Args:
        run_params: The run parameters containing model_size.
        training_modalities: List of sensor labels to support.

    Returns:
        A LatentPredictorConfig for building the model.
    """
    model_size = MODEL_SIZE_ARGS[run_params.model_size]
    return LatentPredictorConfig(
        encoder=EncoderConfig(
            embed_dim=int(model_size["encoder_embedding_size"]),
            num_heads=int(model_size["encoder_num_heads"]),
            depth=int(model_size["encoder_depth"]),
            ffn_expansion=float(model_size["mlp_ratio"]),
            sensor_labels=training_modalities,
        ),
        decoder_depth=int(model_size["decoder_depth"]),
        decoder_num_heads=int(model_size["decoder_num_heads"]),
    )


@dataclass
class ThroughputBenchmarkRunnerConfig(Config):
    """Defines the configuration for a throughput benchmarking run."""

    sweep_dict: dict[str, Any] | None = None
    sweep_keys: list[str] | None = None
    sweep_group_name: str | None = None
    training_modalities: list[str] = field(
        default_factory=lambda: [
            SENTINEL2_L2A.label,
            SENTINEL1.label,
            LANDSAT.label,
        ]
    )
    work_dir: Path = Path("./test_work_dir")
    default_run_params: RunParams | None = None
    save_folder: Path | None = None
    cross_product_sweep: bool = False

    def build(
        self,
        model_config: Any | None = None,
    ) -> "ThroughputBenchmarkRunner":
        """Builds a throughput benchmarking runner.

        Args:
            model_config: Optional pre-built model config. If provided, this config
                will be used for all benchmark runs instead of building one from
                default parameters.
        """
        if self.default_run_params is None:
            self.default_run_params = RunParams()

        if self.sweep_dict is None and self.sweep_keys is None:
            raise ValueError("Either sweep_dict or sweep_keys must be set")
        if self.sweep_dict is not None and self.sweep_keys is not None:
            raise ValueError("Only one of sweep_dict or sweep_keys can be set")

        if self.sweep_dict is not None:
            sweep_dict = self.sweep_dict
        else:
            assert self.sweep_keys is not None
            sweep_dict = {}
            for sweep_key in self.sweep_keys:
                sweep_dict[sweep_key] = constants.SWEEPS[sweep_key]

        return ThroughputBenchmarkRunner(
            default_run_params=self.default_run_params,
            sweep_group_name=self.sweep_group_name,
            training_modalities=self.training_modalities,
            work_dir=self.work_dir,
            save_folder=self.save_folder,
            sweep_dict=sweep_dict,
            cross_product_sweep=self.cross_product_sweep,
            model_config=model_config,
        )


class ThroughputBenchmarkRunner:
    """Runner for a throughput benchmarking run."""

    def __init__(
        self,
        default_run_params: RunParams,
        sweep_group_name: str | None,
        training_modalities: list[str],
        work_dir: Path,
        save_folder: Path | None = None,
        sweep_dict: dict[str, Any] = {},
        cross_product_sweep: bool = False,
        model_config: Any | None = None,
    ):
        """Initializes the throughput benchmarking runner."""
        self.default_run_params = default_run_params
        self.sweep_group_name = sweep_group_name
        self.training_modalities = training_modalities
        self.work_dir = work_dir
        self.work_dir.mkdir(exist_ok=True)
        self.save_folder = save_folder
        self.sweep_dict = sweep_dict
        self.cross_product_sweep = cross_product_sweep
        self.model_config = model_config
        uuid_str = str(uuid.uuid4())[:6]
        self.sweep_name = "_".join(self.sweep_dict.keys()) + "-" + uuid_str

    def build_model(self, run_params: RunParams) -> SpaceNitEncoder:
        """Builds a model based on the run parameters."""
        if self.model_config is not None:
            model_config = self.model_config
        else:
            model_config = build_default_model_config(
                run_params, self.training_modalities
            )
        return SpaceNitEncoder(model_config=model_config)

    def build_sweep_run_params(self) -> list[RunParams]:
        """Builds a list of run parameters based on the sweep dictionary."""
        run_params_list: list[RunParams] = []
        if self.cross_product_sweep:
            sweep_dict_keys = list(self.sweep_dict.keys())
            for combination in itertools.product(
                *[self.sweep_dict[key] for key in sweep_dict_keys]
            ):
                run_params_list.append(
                    self.default_run_params.replace(
                        **dict(zip(sweep_dict_keys, combination))
                    )
                )
        else:
            for key, value in self.sweep_dict.items():
                for v in value:
                    run_params_list.append(self.default_run_params.replace(**{key: v}))
        run_params_list.append(self.default_run_params)
        return run_params_list

    def run_benchmarking_sweep(self, run_params_list: list[RunParams]) -> None:
        """Runs the benchmarking code for a list of run parameters."""
        for run_params in run_params_list:
            try:
                logger.info(f"Running benchmarking for {run_params}")
                self.run_benchmarking(run_params)
            except Exception as e:
                logger.error(f"Error running benchmarking for {run_params}: {e}")
                continue

    def run_benchmarking(self, run_params: RunParams) -> None:
        """Runs the benchmarking code."""
        model = self.build_model(run_params)
        if torch.cuda.is_available() and run_params.gpu_type == "cuda":
            logger.info("Model loaded and on gpu")
            model.to(run_params.gpu_type)
        device = next(model.parameters()).device
        batch_size = run_params.batch_size

        if run_params.bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        callbacks = []
        if run_params.profiler_enabled:
            profiler = ProfilerCallback(
                skip_first=0,
                wait=0,
                warmup=5,
                active=5,
                repeat=1,
            )
            profiler.trainer = MinimalTrainer(device, self.work_dir)
            callbacks.append(profiler)

        if run_params.wandb_enabled:
            project = os.getenv(constants.PARAM_KEYS["project"], constants.PROJECT_NAME)
            owner = os.getenv(constants.PARAM_KEYS["owner"], constants.ENTITY_NAME)
            name = f"{run_params.run_name}-{self.sweep_name}"
            group = self.sweep_group_name
            wandb_callback = WandBCallback(
                project=project,
                entity=owner,
                name=name,
                group=group,
                config=run_params.as_dict(),
            )
            wandb_callback.trainer = MinimalTrainer(device, self.work_dir)
            callbacks.append(wandb_callback)

        for callback in callbacks:
            callback.pre_train()

        # Build synthetic sensor data dict
        sensor_data: dict[str, torch.Tensor] = {}
        if run_params.use_s2:
            sensor_data[SENTINEL2_L2A.label] = torch.rand(
                batch_size,
                run_params.image_size,
                run_params.image_size,
                run_params.num_timesteps,
                NUM_S2_BANDS,
                device=device,
                dtype=dtype,
            )
        if run_params.use_s1:
            sensor_data[SENTINEL1.label] = torch.rand(
                batch_size,
                run_params.image_size,
                run_params.image_size,
                run_params.num_timesteps,
                NUM_S1_BANDS,
                device=device,
                dtype=dtype,
            )
        if run_params.use_landsat:
            sensor_data[LANDSAT.label] = torch.rand(
                batch_size,
                run_params.image_size,
                run_params.image_size,
                run_params.num_timesteps,
                NUM_LANDSAT_BANDS,
                device=device,
                dtype=dtype,
            )

        total_tokens_per_batch = sum(
            t.shape[0] * t.shape[1] * t.shape[2] * t.shape[3]
            for t in sensor_data.values()
        )

        tokens_processed_per_batch: list[int] = []
        time_taken_per_batch: list[float] = []
        logger.info("Data prepared, starting warmup")
        torch.cuda.set_sync_debug_mode("warn")
        oom_occurred = False
        for _ in range(5):
            try:
                with torch.inference_mode():
                    if run_params.bf16:
                        with torch.amp.autocast(
                            device_type=device.type, dtype=torch.bfloat16
                        ):
                            encoded, sensor_ids, layout = model.forward(
                                sensor_data, patch_size=run_params.patch_size
                            )
                    else:
                        encoded, sensor_ids, layout = model.forward(
                            sensor_data, patch_size=run_params.patch_size
                        )
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM during warmup: {e}")
                oom_occurred = True
                break

        if oom_occurred:
            logger.info("CUDA OOM occurred during warmup, skipping benchmark")
            metrics_oom_occurred: dict[str, Any] = {
                constants.OOM_OCCURRED_METRIC: 1,
            }
            for callback in callbacks:
                callback.log_metrics(step=0, metrics=metrics_oom_occurred)
            for callback in callbacks:
                callback.post_train()
            return

        logger.info("Warmup complete, starting benchmark")
        if device.type == "cuda":
            torch.cuda.synchronize()
        overall_start_time = time.monotonic()
        interval_start_time = time.monotonic()
        idx = 0
        while (
            time.monotonic() - interval_start_time
        ) < run_params.benchmark_interval_s or len(
            tokens_processed_per_batch
        ) < run_params.min_batches_per_interval:
            batch_start = time.monotonic()

            with torch.inference_mode():
                if run_params.bf16:
                    with torch.amp.autocast(
                        device_type=device.type, dtype=torch.bfloat16
                    ):
                        encoded, sensor_ids, layout = model.forward(
                            sensor_data,
                            patch_size=run_params.patch_size,
                        )
                else:
                    encoded, sensor_ids, layout = model.forward(
                        sensor_data, patch_size=run_params.patch_size
                    )
            time_taken_per_batch.append(time.monotonic() - batch_start)

            for callback in callbacks:
                callback.pre_load_batch()

            # Count tokens from the layout
            num_tokens = sum(count for _, count in layout) * batch_size
            tokens_processed_per_batch.append(num_tokens)

        if device.type == "cuda":
            torch.cuda.synchronize()
        overall_time_taken = time.monotonic() - overall_start_time
        logger.info(
            f"Overall time taken: {overall_time_taken} sum of time taken per batch: "
            f"{sum(time_taken_per_batch)} num batches: {len(time_taken_per_batch)}"
        )
        metrics_to_submit: dict[str, Any] = {
            constants.PER_BATCH_TOKEN_RATE_METRIC: wandb.Histogram(
                np.array(
                    [
                        tokens_processed_per_batch,
                        time_taken_per_batch,
                    ]
                )
            ),
            constants.MEAN_BATCH_TOKEN_RATE_METRIC: sum(tokens_processed_per_batch)
            / overall_time_taken,
            constants.MEAN_BATCH_TIME_METRIC: overall_time_taken
            / len(time_taken_per_batch),
            constants.NUM_TOKENS_PER_BATCH_METRIC: sum(tokens_processed_per_batch)
            / len(tokens_processed_per_batch),
        }
        num_batches = len(time_taken_per_batch)
        num_centroids = num_batches * batch_size
        centroids_per_second = num_centroids / overall_time_taken
        tile_km2 = (
            run_params.image_size * REFERENCE_GROUND_RESOLUTION / 1000.0
        ) ** 2
        area_processed_km2 = batch_size * tile_km2 * num_batches
        square_km_per_second = area_processed_km2 / overall_time_taken
        metrics_to_submit[constants.SQUARE_KM_PER_SECOND_METRIC] = square_km_per_second
        metrics_to_submit[constants.PIXELS_PER_SECOND_METRIC] = centroids_per_second
        try:
            gpu_name = torch.cuda.get_device_name(device)
            metrics_to_submit[constants.GPU_NAME_METRIC] = gpu_name
        except Exception as e:
            logger.error(f"Error getting GPU name: {e}")

        logger.info(f"Metrics for {batch_size} were: {metrics_to_submit}")
        for callback in callbacks:
            callback.log_metrics(step=idx, metrics=metrics_to_submit)
        for callback in callbacks:
            callback.post_train()

    def run(self) -> None:
        """Runs the throughput benchmarking."""
        run_params_list = self.build_sweep_run_params()
        logger.info(
            f"Running {len(run_params_list)} benchmarking runs sweeping over {self.sweep_dict}"
        )
        self.run_benchmarking_sweep(run_params_list)
