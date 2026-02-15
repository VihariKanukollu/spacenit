"""SpaceNit experiment logger (WandB) hook."""

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from olmo_core.distributed.utils import get_rank
from olmo_core.exceptions import OLMoEnvironmentError
from olmo_core.train.callbacks.wandb import WANDB_API_KEY_ENV_VAR, WandBCallback
from tqdm import tqdm

from spacenit.ingestion.sensors import SensorRegistry
from spacenit.ingestion.tile_loader import GeoTileLoader
from spacenit.ingestion.tile_dataset import GeoTileDataset, GetItemArgs
from spacenit.ingestion.rendering import (
    plot_latlon_distribution,
    plot_modality_data_distribution,
)

logger = logging.getLogger(__name__)


# Default tile size constant (matching the standard 10 m/px tile dimension)
_IMAGE_TILE_SIZE = 128


def get_sample_data_for_histogram(
    dataset: GeoTileDataset, num_samples: int = 100, num_values: int = 100
) -> dict[str, Any]:
    """Get the sample data per sensor per band for showing the histogram.

    Args:
        dataset: The dataset to sample from.
        num_samples: The number of samples to sample from the dataset.
        num_values: The number of values to sample from each sensor per band.

    Returns:
        dict: A dictionary containing the sample data per sensor per band.
    """
    if num_samples > len(dataset):
        raise ValueError(
            f"num_samples {num_samples} is greater than the number of samples in the dataset {len(dataset)}"
        )
    indices_to_sample = random.sample(list(range(len(dataset))), k=num_samples)
    sample_data: dict[str, Any] = {}

    for i in tqdm(indices_to_sample):
        get_item_args = GetItemArgs(idx=i, patch_size=1, sampled_hw_p=_IMAGE_TILE_SIZE)
        _, sample = dataset[get_item_args]
        for sensor_name in sample.present_keys:
            if sensor_name == "timestamps" or sensor_name == "latlon":
                continue
            sensor_data = sample.to_dict(ignore_nones=True)[sensor_name]
            if sensor_data is None:
                continue
            sensor_spec = SensorRegistry.get(sensor_name)
            sensor_bands = sensor_spec.all_channel_names
            if sensor_name not in sample_data:
                sample_data[sensor_name] = {band: [] for band in sensor_bands}
            for idx, band in enumerate(sensor_bands):
                sample_data[sensor_name][band].extend(
                    random.sample(
                        sensor_data[:, :, :, idx].flatten().tolist(), num_values
                    )
                )
    return sample_data


@dataclass
class SpaceNitExperimentLogger(WandBCallback):
    """SpaceNit experiment logger (WandB) hook."""

    upload_dataset_distribution_pre_train: bool = True
    upload_modality_data_band_distribution_pre_train: bool = False
    restart_on_same_run: bool = True

    def pre_train(self) -> None:
        """Pre-train hook for the experiment logger."""
        if self.enabled and get_rank() == 0:
            if WANDB_API_KEY_ENV_VAR not in os.environ:
                raise OLMoEnvironmentError(f"missing env var '{WANDB_API_KEY_ENV_VAR}'")

            wandb_dir = Path(self.trainer.save_folder) / "wandb"
            wandb_dir.mkdir(parents=True, exist_ok=True)
            resume_id = None
            if self.restart_on_same_run:
                runid_file = wandb_dir / "wandb_runid.txt"
                if runid_file.exists():
                    resume_id = runid_file.read_text().strip()

            self.wandb.init(
                dir=wandb_dir,
                project=self.project,
                entity=self.entity,
                group=self.group,
                name=self.name,
                tags=self.tags,
                notes=self.notes,
                config=self.config,
                id=resume_id,
                resume="allow",
                settings=self.wandb.Settings(init_timeout=240),
            )

            if not resume_id and self.restart_on_same_run:
                runid_file.write_text(self.run.id)

            self._run_path = self.run.path  # type: ignore
            if self.upload_dataset_distribution_pre_train:
                assert isinstance(self.trainer.data_loader, GeoTileLoader)
                dataset = self.trainer.data_loader.dataset
                logger.info("Gathering locations of entire dataset")
                latlons = dataset.latlon_distribution
                assert latlons is not None
                logger.info(f"Uploading dataset distribution to wandb: {latlons.shape}")
                fig = plot_latlon_distribution(
                    latlons, "Geographic Distribution of Dataset"
                )
                self.wandb.log(
                    {
                        "dataset/pretraining_geographic_distribution": self.wandb.Image(
                            fig
                        )
                    }
                )
                plt.close(fig)
                # Delete the latlon distribution from the dataset so it doesn't get pickled into data worker processes
                del dataset.latlon_distribution
                if self.upload_modality_data_band_distribution_pre_train:
                    logger.info("Gathering normalized data distribution")
                    sample_data = get_sample_data_for_histogram(dataset)
                    for sensor_name, sensor_data in sample_data.items():
                        fig = plot_modality_data_distribution(sensor_name, sensor_data)
                        self.wandb.log(
                            {
                                f"dataset/pretraining_{sensor_name}_distribution": self.wandb.Image(
                                    fig
                                )
                            }
                        )
                        plt.close(fig)
