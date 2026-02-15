"""Code for configuring and running SpaceNit experiments."""

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from olmo_core.config import StrEnum
from olmo_core.distributed.utils import get_local_rank
from olmo_core.launch.beaker import BeakerLaunchConfig, ExperimentSpec
from olmo_core.train import (
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import ConfigSaverCallback, WandBCallback
from olmo_core.utils import get_default_device, prepare_cli_environment, seed_all

from spacenit.settings import Config
from spacenit.ingestion.sensors import SensorRegistry
from spacenit.ingestion.tile_loader import GeoTileLoaderConfig
from spacenit.ingestion.merged_dataset import GeoTileMergedDatasetConfig
from spacenit.ingestion.rendering import visualize_sample
from spacenit.profiling.throughput_runner import ThroughputBenchmarkRunnerConfig
from spacenit.ops.helpers import (
    MockLatentMIMTrainModule,
    MockGeoTileLoader,
)
from spacenit.arch.band_tokenization import TokenizationConfig
from spacenit.pipeline.runners import SpaceNitTrainRunnerConfig

logger = logging.getLogger(__name__)


@dataclass
class SpaceNitBeakerLaunchConfig(BeakerLaunchConfig):
    """Extend BeakerLaunchConfig with hostnames option.

    This enables targeting specific Beaker hosts.
    """

    hostnames: list[str] | None = None

    def build_experiment_spec(
        self, torchrun: bool = True, entrypoint: str | None = None
    ) -> ExperimentSpec:
        """Build the experiment spec."""
        spec = super().build_experiment_spec(torchrun, entrypoint)
        if self.hostnames:
            constraints = spec.tasks[0].constraints
            constraints.cluster = None
            constraints.hostname = self.hostnames
        return spec


@dataclass
class CommonComponents(Config):
    """Any configurable items that are common to all experiments.

    Args:
        run_name: Name of the experiment run.
        save_folder: Path to save checkpoints and logs.
        training_modalities: List of modality names to train on.
        launch: Optional Beaker launch configuration.
        nccl_debug: Whether to enable NCCL debugging.
        tokenization_config: Optional custom tokenization config for band groupings.
    """

    run_name: str
    save_folder: str
    training_modalities: list[str]
    launch: SpaceNitBeakerLaunchConfig | None = None
    nccl_debug: bool = False
    tokenization_config: TokenizationConfig | None = None

    def validate(self) -> None:
        """Validate the common components."""
        if not isinstance(self.training_modalities, list):
            raise ValueError("training_modalities must be a list")
        if not all(
            modality in SensorRegistry.all_labels() for modality in self.training_modalities
        ):
            raise ValueError(
                "training_modalities must contain only valid modality names"
            )
        if self.tokenization_config is not None:
            self.tokenization_config.validate()


@dataclass
class SpaceNitVisualizeConfig(Config):
    """Configuration for visualizing the dataset."""

    output_dir: str
    num_samples: int | None = None
    global_step: int | None = None
    std_multiplier: float = 2.0


@dataclass
class SpaceNitExperimentConfig(Config):
    """Configuration for a SpaceNit experiment."""

    run_name: str
    model: Config
    dataset: Config
    data_loader: GeoTileLoaderConfig
    train_module: SpaceNitTrainRunnerConfig
    trainer: TrainerConfig
    launch: SpaceNitBeakerLaunchConfig | None = None
    visualize: SpaceNitVisualizeConfig | None = None
    init_seed: int = 12536


@dataclass
class SpaceNitEvaluateConfig(Config):
    """Configuration for a SpaceNit evaluation experiment."""

    run_name: str
    model: Config
    trainer: TrainerConfig
    launch: SpaceNitBeakerLaunchConfig | None = None
    train_module: SpaceNitTrainRunnerConfig | None = None
    init_seed: int = 12536


@dataclass
class BenchmarkExperimentConfig(Config):
    """Configuration for a throughput benchmarking run."""

    benchmark: ThroughputBenchmarkRunnerConfig
    model: Config | None = None
    launch: SpaceNitBeakerLaunchConfig | None = None


def split_common_overrides(overrides: list[str]) -> tuple[list[str], list[str]]:
    """Split the common overrides from the command line."""
    common_overrides = [
        dotfield.replace("common.", "")
        for dotfield in overrides
        if "common." in dotfield
    ]
    non_common_overrides = [
        dotfield for dotfield in overrides if "common." not in dotfield
    ]
    return common_overrides, non_common_overrides


def build_config(
    common: CommonComponents,
    model_config_builder: Callable[[CommonComponents], Config],
    dataset_config_builder: Callable[[CommonComponents], GeoTileMergedDatasetConfig],
    dataloader_config_builder: Callable[[CommonComponents], GeoTileLoaderConfig],
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig],
    train_module_config_builder: Callable[
        [CommonComponents],
        SpaceNitTrainRunnerConfig,
    ],
    overrides: list[str],
    visualize_config_builder: (
        Callable[[CommonComponents], SpaceNitVisualizeConfig] | None
    ) = None,
) -> SpaceNitExperimentConfig:
    """Build a SpaceNit experiment configuration."""
    common_overrides, overrides = split_common_overrides(overrides)
    logger.info("Common overrides: %s", common_overrides)
    common = common.merge(common_overrides)
    logger.info("Common: %s", common)
    model_config = model_config_builder(common)
    dataset_config = dataset_config_builder(common)
    dataloader_config = dataloader_config_builder(common)
    trainer_config = trainer_config_builder(common)
    train_module_config = train_module_config_builder(common)
    visualize_config = (
        visualize_config_builder(common) if visualize_config_builder else None
    )
    config = SpaceNitExperimentConfig(
        run_name=common.run_name,
        model=model_config,
        dataset=dataset_config,
        data_loader=dataloader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        visualize=visualize_config,
        launch=common.launch,
    )
    logger.info("Overrides: %s", overrides)
    config = config.merge(overrides)
    return config


def build_evaluate_config(
    common: CommonComponents,
    model_config_builder: Callable[[CommonComponents], Config],
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig],
    overrides: list[str],
    train_module_config_builder: (
        Callable[[CommonComponents], SpaceNitTrainRunnerConfig] | None
    ) = None,
) -> SpaceNitEvaluateConfig:
    """Build a SpaceNit evaluation experiment configuration."""
    common_overrides, overrides = split_common_overrides(overrides)
    logger.info("Common overrides: %s", common_overrides)
    common = common.merge(common_overrides)
    logger.info("Common: %s", common)
    model_config = model_config_builder(common)
    trainer_config = trainer_config_builder(common)
    train_module_config = (
        train_module_config_builder(common) if train_module_config_builder else None
    )
    config = SpaceNitEvaluateConfig(
        run_name=common.run_name,
        model=model_config,
        trainer=trainer_config,
        launch=common.launch,
        train_module=train_module_config,
    )
    config = config.merge(overrides)
    return config


def build_benchmark_config(
    common: CommonComponents,
    inference_benchmarking_config_builder: Callable[
        [CommonComponents], ThroughputBenchmarkRunnerConfig
    ],
    overrides: list[str],
    benchmark_model_config_builder: Callable[[CommonComponents], Config] | None = None,
) -> BenchmarkExperimentConfig:
    """Build a throughput benchmarking configuration."""
    inference_benchmarking_config = inference_benchmarking_config_builder(common)

    model_config = None
    if benchmark_model_config_builder is not None:
        model_config = benchmark_model_config_builder(common)

    config = BenchmarkExperimentConfig(
        launch=common.launch,
        benchmark=inference_benchmarking_config,
        model=model_config,
    )
    config = config.merge(overrides)
    logger.info("Benchmark config: %s", config)
    return config


def benchmark(config: BenchmarkExperimentConfig) -> None:
    """Benchmark an experiment."""
    runner = config.benchmark.build(model_config=config.model)
    runner.run()


def launch_benchmark(config: BenchmarkExperimentConfig) -> None:
    """Launch a throughput benchmarking run."""
    assert config.launch is not None
    config.launch.launch(follow=False, torchrun=False)


def train(config: SpaceNitExperimentConfig) -> None:
    """Train an experiment."""
    seed_all(config.init_seed)

    model = config.model.build()
    device = get_default_device()
    model = model.to(device)
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(
        dataset,
        dp_process_group=train_module.dp_process_group,
    )
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict
    trainer.fit()


def evaluate(config: SpaceNitEvaluateConfig) -> None:
    """Evaluate a checkpoint or model on downstream tasks."""
    seed_all(config.init_seed)

    model = config.model.build()
    device = get_default_device()
    model = model.to(device)
    data_loader = MockGeoTileLoader()

    if config.trainer.load_path is not None:
        if config.train_module is None:
            raise ValueError("train_module is not set so we can't load the checkpoint")
        train_module = config.train_module.build(model)
        data_loader.min_patch_size = model.encoder.min_patch_size
        data_loader.max_patch_size = model.encoder.max_patch_size
    else:
        train_module = MockLatentMIMTrainModule()

    train_module.model = model
    trainer = config.trainer.build(train_module, data_loader)
    config_dict = config.as_config_dict()
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict
    trainer.fit()


def visualize(config: SpaceNitExperimentConfig) -> None:
    """Visualize the dataset for an experiment."""
    logger.info("Visualizing the dataset")
    if config.visualize is None:
        raise ValueError("visualize_config is not set")
    global_step = config.visualize.global_step
    dataset = config.dataset.build()
    if global_step is not None:
        data_loader = config.data_loader.build(dataset, dp_process_group=None)
        sample_indices = data_loader.fast_forward(global_step)
    else:
        sample_indices = np.random.randint(
            0, len(dataset), config.visualize.num_samples
        )
    logger.info(f"sample indices: {sample_indices}")
    for sample_index in sample_indices:
        visualize_sample(dataset, sample_index, config.visualize.output_dir)
    logger.info("Done visualizing the dataset")


def launch(config: SpaceNitExperimentConfig) -> None:
    """Launch an experiment."""
    logger.info("Launching the experiment")
    logger.info(config)
    assert config.launch is not None
    config.launch.launch(follow=False, torchrun=True)


def prep(config: SpaceNitExperimentConfig) -> None:
    """Prepare the dataset for an experiment."""
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=None)
    data_loader.reshuffle(epoch=1)


def launch_prep(config: SpaceNitExperimentConfig) -> None:
    """Launch the preparation of the dataset for an experiment."""
    assert config.launch is not None
    config.launch.num_gpus = 0
    config.launch.num_nodes = 1
    logger.info(config)
    logger.info("Launching the preparation of the dataset...")
    config.launch.launch(follow=True, torchrun=False)


class SubCmd(StrEnum):
    """Subcommands for SpaceNit experiments."""

    launch = "launch"
    train = "train"
    train_single = "train_single"
    evaluate = "evaluate"
    launch_evaluate = "launch_evaluate"
    prep = "prep"
    launch_prep = "launch_prep"
    dry_run = "dry_run"
    dry_run_evaluate = "dry_run_evaluate"
    visualize = "visualize"
    benchmark = "benchmark"
    launch_benchmark = "launch_benchmark"

    def prepare_environment(self) -> None:
        """Prepare the environment for the given subcommand."""
        if self in (
            SubCmd.launch,
            SubCmd.dry_run,
            SubCmd.dry_run_evaluate,
            SubCmd.prep,
            SubCmd.launch_prep,
            SubCmd.visualize,
            SubCmd.benchmark,
            SubCmd.launch_benchmark,
            SubCmd.launch_evaluate,
        ):
            prepare_cli_environment()
        elif self == SubCmd.train or self == SubCmd.evaluate:
            prepare_training_environment()
        elif self == SubCmd.train_single:
            prepare_training_environment(backend=None)
        else:
            raise NotImplementedError(self)

    def run(
        self,
        config: SpaceNitExperimentConfig | BenchmarkExperimentConfig,
    ) -> None:
        """Run the given subcommand."""
        if get_local_rank() == 0:
            print(config)

        if self == SubCmd.launch or self == SubCmd.launch_evaluate:
            launch(config)
        elif self == SubCmd.dry_run or self == SubCmd.dry_run_evaluate:
            logger.info(config)
        elif self == SubCmd.visualize:
            seed_all(config.init_seed)
            visualize(config)
        elif self == SubCmd.train:
            try:
                train(config)
            finally:
                teardown_training_environment()
        elif self == SubCmd.evaluate:
            try:
                evaluate(config)
            finally:
                teardown_training_environment()
        elif self == SubCmd.train_single:
            if config.train_module.dp_config is not None:
                logger.warning(
                    "'dp_config' is set to %s, but you can't use data parallelism when running on a single node. Disabling.",
                    config.train_module.dp_config,
                )
                config.train_module.dp_config = None
            try:
                train(config)
            finally:
                teardown_training_environment()
        elif self == SubCmd.prep:
            prep(config)
        elif self == SubCmd.launch_prep:
            launch_prep(config)
        elif self == SubCmd.benchmark:
            benchmark(config)
        elif self == SubCmd.launch_benchmark:
            launch_benchmark(config)
        else:
            raise NotImplementedError(self)


def main(
    *,
    common_components_builder: Callable,
    model_config_builder: Callable[[CommonComponents], Config] | None = None,
    dataset_config_builder: Callable[[CommonComponents], Config] | None = None,
    dataloader_config_builder: (
        Callable[[CommonComponents], GeoTileLoaderConfig] | None
    ) = None,
    trainer_config_builder: Callable[[CommonComponents], TrainerConfig] | None = None,
    train_module_config_builder: (
        Callable[[CommonComponents], SpaceNitTrainRunnerConfig] | None
    ) = None,
    visualize_config_builder: (
        Callable[[CommonComponents], SpaceNitVisualizeConfig] | None
    ) = None,
    inference_benchmarking_config_builder: (
        Callable[[CommonComponents], ThroughputBenchmarkRunnerConfig] | None
    ) = None,
    benchmark_model_config_builder: Callable[[CommonComponents], Config] | None = None,
) -> None:
    """Main entry point for SpaceNit experiments."""
    usage = f"""
[yellow]Usage:[/] [i blue]python[/] [i cyan]{sys.argv[0]}[/] [i b magenta]{"|".join(SubCmd)}[/] [i b]RUN_NAME CLUSTER[/] [i][OVERRIDES...][/]
If running command on a local machine, you can use the [b]local[/b] cluster name.
    """.strip()
    logger.info(f"Running {sys.argv}")
    if len(sys.argv) < 4 or sys.argv[1] not in set(SubCmd):
        import rich

        rich.get_console().print(usage, highlight=False)
        sys.exit(1)

    script, cmd, run_name, cluster, *overrides = sys.argv
    common = common_components_builder(script, cmd, run_name, cluster, overrides)

    cmd = SubCmd(cmd)
    cmd.prepare_environment()

    if cmd == SubCmd.benchmark or cmd == SubCmd.launch_benchmark:
        if inference_benchmarking_config_builder is None:
            raise ValueError("inference_benchmarking_config_builder is not set")
        config = build_benchmark_config(
            common=common,
            inference_benchmarking_config_builder=inference_benchmarking_config_builder,
            overrides=overrides,
            benchmark_model_config_builder=benchmark_model_config_builder,
        )
    elif (
        cmd == SubCmd.evaluate
        or cmd == SubCmd.launch_evaluate
        or cmd == SubCmd.dry_run_evaluate
    ):
        assert model_config_builder is not None
        assert trainer_config_builder is not None
        config = build_evaluate_config(
            common=common,
            model_config_builder=model_config_builder,
            trainer_config_builder=trainer_config_builder,
            overrides=overrides,
            train_module_config_builder=train_module_config_builder,
        )
    else:
        assert model_config_builder is not None
        assert dataset_config_builder is not None
        assert dataloader_config_builder is not None
        assert trainer_config_builder is not None
        assert train_module_config_builder is not None
        config = build_config(
            common=common,
            model_config_builder=model_config_builder,
            dataset_config_builder=dataset_config_builder,
            dataloader_config_builder=dataloader_config_builder,
            trainer_config_builder=trainer_config_builder,
            train_module_config_builder=train_module_config_builder,
            visualize_config_builder=visualize_config_builder,
            overrides=overrides,
        )

    cmd.run(config)
