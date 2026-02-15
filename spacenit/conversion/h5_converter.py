"""Convert a directory of GeoTIFF tiles into an HDF5 training dataset.

The :class:`GeoTileH5Writer` reads parsed sensor tiles, assembles training
samples, applies quality filters, and writes each sample to a self-contained
HDF5 file.  Compression, chunking, and sub-tiling are all configurable.
"""

import json
import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from typing import Any

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from einops import rearrange
from tqdm import tqdm
from upath import UPath

from spacenit.settings import Config
from spacenit.ingestion.sensors import (
    ANNUAL_STEP_COUNT,
    SAR_NODATA_SENTINEL,
    TILE_EDGE_PIXELS,
    SensorSpec,
    TemporalCadence,
    ERA5_10,
    OPENSTREETMAP_RASTER,
    SENTINEL1,
    specs_from_labels,
)

from .csv_parser import parse_tile_dataset
from .tile_sampler import (
    TileSample,
    SensorTile,
    assemble_samples_from_tiles,
    load_raster_for_sample,
)

logger = logging.getLogger(__name__)

mp.set_start_method("spawn", force=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _amplitude_to_decibels(data: np.ndarray) -> np.ndarray:
    """Convert linear amplitude data to decibels.

    Args:
        data: linear-scale array.

    Returns:
        Array in dB units.
    """
    data = np.clip(data, 1e-10, None)
    return 10 * np.log10(data)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class GeoTileH5WriterConfig(Config):
    """Configuration for :class:`GeoTileH5Writer`.

    See https://docs.h5py.org/en/stable/high/dataset.html for compression
    settings.
    """

    tile_path: str
    supported_sensor_labels: list[str]
    multiprocessed_h5_creation: bool = True
    compression: str | None = None
    compression_opts: int | None = None
    shuffle: bool | None = None
    chunk_options: tuple | None = None
    tile_size: int = TILE_EDGE_PIXELS
    reserved_cores: int = 10
    required_sensor_labels: list[str] = field(default_factory=list)

    def build(self) -> "GeoTileH5Writer":
        """Construct the :class:`GeoTileH5Writer` from this config."""
        return GeoTileH5Writer(
            tile_path=UPath(self.tile_path),
            supported_sensors=specs_from_labels(self.supported_sensor_labels),
            multiprocessed_h5_creation=self.multiprocessed_h5_creation,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle,
            chunk_options=self.chunk_options,
            tile_size=self.tile_size,
            reserved_cores=self.reserved_cores,
            required_sensors=specs_from_labels(self.required_sensor_labels),
        )


# ---------------------------------------------------------------------------
# GeoTileH5Writer
# ---------------------------------------------------------------------------


class GeoTileH5Writer:
    """Converts a directory of GeoTIFF tiles into per-sample HDF5 files."""

    h5_folder: str = "h5py_data_w_missing_timesteps"
    latlon_distribution_fname: str = "latlon_distribution.npy"
    sample_metadata_fname: str = "sample_metadata.csv"
    sample_file_pattern: str = "sample_{index}.h5"
    compression_settings_fname: str = "compression_settings.json"
    missing_timesteps_mask_group: str = "missing_timesteps_masks"

    def __init__(
        self,
        tile_path: UPath,
        supported_sensors: list[SensorSpec],
        multiprocessed_sample_processing: bool = True,
        multiprocessed_h5_creation: bool = True,
        compression: str | None = None,
        compression_opts: int | None = None,
        shuffle: bool | None = None,
        chunk_options: tuple | bool | None = None,
        tile_size: int = TILE_EDGE_PIXELS,
        reserved_cores: int = 10,
        required_sensors: list[SensorSpec] | None = None,
    ) -> None:
        """Initialise the writer.

        Args:
            tile_path: Root directory containing manifest CSVs and tile data.
            supported_sensors: Sensors to include in the HDF5 dataset.
            multiprocessed_sample_processing: Process samples in parallel.
            multiprocessed_h5_creation: Create HDF5 files in parallel.
            compression: Compression algorithm (``None``, ``"gzip"``,
                ``"zstd"``, ``"lz4"``).
            compression_opts: Compression level (0–9 for gzip).
            shuffle: Enable the HDF5 shuffle filter.
            chunk_options: Chunking configuration.  ``None`` disables chunking,
                ``True`` enables auto-chunking, and a tuple specifies a fixed
                chunk shape.
            tile_size: Edge length of the sub-tile written to each sample.
            reserved_cores: CPU cores to keep free during multiprocessing.
            required_sensors: Samples missing any of these sensors are dropped.
        """
        self.tile_path = tile_path
        self.supported_sensors = supported_sensors
        logger.info(f"Supported sensors: {self.supported_sensors}")
        self.multiprocessed_sample_processing = multiprocessed_sample_processing
        self.multiprocessed_h5_creation = multiprocessed_h5_creation
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.chunk_options = chunk_options
        self.h5_dir: UPath | None = None
        self.required_sensors = required_sensors or []

        if TILE_EDGE_PIXELS % tile_size != 0:
            raise ValueError(
                f"tile_size {tile_size} must be a factor of {TILE_EDGE_PIXELS}"
            )
        self.tile_size = tile_size
        self.subtiles_per_dim = TILE_EDGE_PIXELS // tile_size
        self.total_subtiles = self.subtiles_per_dim ** 2
        self.reserved_cores = reserved_cores

    # ------------------------------------------------------------------
    # Naming helpers
    # ------------------------------------------------------------------

    @property
    def compression_suffix(self) -> str:
        """String suffix encoding compression settings (for folder names)."""
        parts = ""
        if self.compression is not None:
            parts = f"_{self.compression}"
        if self.compression_opts is not None:
            parts += f"_{self.compression_opts}"
        if self.shuffle is not None:
            parts += "_shuffle"
        return parts

    @property
    def tile_size_suffix(self) -> str:
        """String suffix encoding tile geometry."""
        return f"_{self.tile_size}_x_{self.total_subtiles}"

    # ------------------------------------------------------------------
    # Sample retrieval
    # ------------------------------------------------------------------

    def _retrieve_samples(self) -> list[TileSample]:
        """Parse the raw tile directory and assemble training samples."""
        tiles = parse_tile_dataset(self.tile_path, self.supported_sensors)
        samples = assemble_samples_from_tiles(tiles, self.supported_sensors)
        logger.info(f"Total samples: {len(samples)}")
        logger.info("Distribution of samples before filtering:\n")
        self._log_sensor_distribution(samples)
        return samples

    # ------------------------------------------------------------------
    # Per-sample HDF5 writing
    # ------------------------------------------------------------------

    def write_sample_h5(
        self, index_sample_tuple: tuple[int, tuple[int, TileSample]]
    ) -> None:
        """Write a single sample to an HDF5 file."""
        i, (subtile_idx, sample) = index_sample_tuple
        h5_path = self._h5_file_path(i)
        self._write_h5_file(sample, h5_path, subtile_idx)

    def create_h5_dataset(
        self, samples: list[tuple[int, TileSample]]
    ) -> None:
        """Write all samples to HDF5 files, optionally in parallel."""
        total = len(samples)

        if self.multiprocessed_h5_creation:
            num_procs = max(1, mp.cpu_count() - self.reserved_cores)
            logger.info(f"Creating H5 dataset using {num_procs} processes")
            with mp.Pool(processes=num_procs) as pool:
                _ = list(
                    tqdm(
                        pool.imap(self.write_sample_h5, enumerate(samples)),
                        total=total,
                        desc="Creating H5 files",
                    )
                )
        else:
            for i, (subtile_idx, sample) in enumerate(samples):
                logger.info(f"Processing sample {i}")
                self.write_sample_h5((i, (subtile_idx, sample)))

    # ------------------------------------------------------------------
    # Metadata persistence
    # ------------------------------------------------------------------

    def save_sample_metadata(
        self, samples: list[tuple[int, TileSample]]
    ) -> None:
        """Write a CSV recording which sensors are present in each sample."""
        if self.h5_dir is None:
            raise ValueError("h5_dir is not set")
        csv_path = self.h5_dir / self.sample_metadata_fname
        logger.info(f"Writing metadata CSV to {csv_path}")

        metadata: dict[str, list] = {"sample_index": []}
        for sensor in self.supported_sensors:
            metadata[sensor.label] = []

        for i, (_, sample) in enumerate(samples):
            metadata["sample_index"].append(i)
            for sensor in self.supported_sensors:
                metadata[sensor.label].append(
                    1 if sensor in sample.sensors else 0
                )

        df = pd.DataFrame(metadata)
        df.to_csv(csv_path, index=False)

    def _h5_file_path(self, index: int) -> UPath:
        """Return the HDF5 path for a given sample index."""
        if self.h5_dir is None:
            raise ValueError("h5_dir is not set")
        return self.h5_dir / self.sample_file_pattern.format(index=index)

    @property
    def latlon_distribution_path(self) -> UPath:
        """Path to the saved lat/lon distribution array."""
        if self.h5_dir is None:
            raise ValueError("h5_dir is not set")
        return self.h5_dir / self.latlon_distribution_fname

    def save_latlon_distribution(
        self, samples: list[tuple[int, TileSample]]
    ) -> None:
        """Persist the lat/lon distribution of all samples."""
        logger.info(f"Saving latlon distribution to {self.latlon_distribution_path}")
        latlons = np.array([sample.compute_latlon() for _, sample in samples])
        with self.latlon_distribution_path.open("wb") as f:
            np.save(f, latlons)

    # ------------------------------------------------------------------
    # Timestamp alignment helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _longest_timestamp_array(
        spacetime_sensors: dict[SensorSpec, np.ndarray],
    ) -> np.ndarray:
        """Return the timestamp array from the sensor with the most steps."""
        return spacetime_sensors[
            max(spacetime_sensors, key=lambda k: len(spacetime_sensors[k]))
        ]

    @staticmethod
    def _build_missing_timestep_masks(
        spacetime_sensors: dict[SensorSpec, np.ndarray],
        reference_timestamps: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Create boolean masks indicating which reference timesteps are present."""
        masks: dict[str, np.ndarray] = {}
        for spec, ts in spacetime_sensors.items():
            mask = np.array(
                [
                    np.any(np.all(ref_ts == ts, axis=1))
                    for ref_ts in reference_timestamps
                ],
                dtype=bool,
            )
            masks[spec.label] = mask
        return masks

    # ------------------------------------------------------------------
    # Quality filtering
    # ------------------------------------------------------------------

    def _remove_bad_sensors_from_sample(
        self, sample: TileSample
    ) -> TileSample:
        """Strip sensors whose raster data fails quality checks."""
        to_remove: set[SensorSpec] = set()
        for sensor in sample.sensors:
            tile = sample.sensors[sensor]
            image = self._load_and_rearrange(tile, sample)

            if np.any(np.isnan(image)):
                logger.warning(
                    f"Sensor {sensor.label} contains NaN values — removing"
                )
                to_remove.add(sensor)

            if sensor == ERA5_10 or sensor == OPENSTREETMAP_RASTER:
                if np.all(image == 0):
                    logger.warning(
                        f"Sensor {sensor.label} is all zeros — removing"
                    )
                    to_remove.add(sensor)

            if sensor == SENTINEL1 and np.any(image == SAR_NODATA_SENTINEL):
                logger.warning(
                    f"Sensor {sensor.label} contains nodata values — removing"
                )
                to_remove.add(sensor)

        for sensor in to_remove:
            del sample.sensors[sensor]
        return sample

    def _process_samples(
        self, samples: list[TileSample]
    ) -> list[TileSample]:
        """Run quality checks on all samples, optionally in parallel."""
        total = len(samples)
        if self.multiprocessed_sample_processing:
            num_procs = max(1, mp.cpu_count() - self.reserved_cores)
            logger.info(f"Processing samples using {num_procs} processes")
            with mp.Pool(processes=num_procs) as pool:
                processed = list(
                    tqdm(
                        pool.imap(self._remove_bad_sensors_from_sample, samples),
                        total=total,
                        desc="Processing samples",
                    )
                )
        else:
            processed = []
            for i, sample in enumerate(samples):
                logger.info(f"Processing sample {i}")
                processed.append(self._remove_bad_sensors_from_sample(sample))
        return processed

    # ------------------------------------------------------------------
    # HDF5 file creation
    # ------------------------------------------------------------------

    def _write_h5_file(
        self,
        sample: TileSample,
        h5_path: UPath,
        subtile_idx: int,
    ) -> dict[str, Any]:
        """Create a single HDF5 file for one sample."""
        sample_dict: dict[str, Any] = {}
        sample_dict["latlon"] = sample.compute_latlon().astype(np.float32)
        timestamp_dict = sample.extract_timestamps()

        # Align timestamps across spacetime-varying sensors.
        spacetime_sensors = {
            sensor: ts
            for sensor, ts in timestamp_dict.items()
            if sensor.varies_in_space_and_time
        }
        reference_timestamps = self._longest_timestamp_array(spacetime_sensors)
        missing_masks = self._build_missing_timestep_masks(
            spacetime_sensors, reference_timestamps
        )
        sample_dict["timestamps"] = reference_timestamps

        # Load raster data for every sensor in the sample.
        for sensor, tile in sample.sensors.items():
            image = self._load_and_rearrange(tile, sample)

            if sensor == SENTINEL1:
                image = _amplitude_to_decibels(image)

            if sensor.varies_in_space:
                if image.shape[0] != image.shape[1]:
                    raise ValueError("Expected square image")
                if image.shape[0] % self.subtiles_per_dim != 0:
                    raise ValueError(
                        f"Image size {image.shape[0]} is not divisible by "
                        f"subtile count {self.subtiles_per_dim}"
                    )
                edge = image.shape[0] // self.subtiles_per_dim
                row = (subtile_idx // self.subtiles_per_dim) * edge
                col = (subtile_idx % self.subtiles_per_dim) * edge
                logger.info(f"Subtile index: {subtile_idx}, row: {row}, col: {col}")
                logger.info(f"Image shape: {image.shape}")
                image = image[row : row + edge, col : col + edge, ...]
                logger.info(f"Image shape after slicing: {image.shape}")

            sample_dict[sensor.label] = image

        # Write to HDF5.
        with h5_path.open("w+b") as f:
            with h5py.File(f, "w") as h5file:
                for item_name, data_item in sample_dict.items():
                    logger.info(
                        f"Writing item {item_name} to h5 file path {h5_path}"
                    )
                    create_kwargs: dict[str, Any] = {}

                    if self.compression is not None:
                        if self.compression == "gzip":
                            create_kwargs["compression"] = self.compression
                            if self.compression_opts is not None:
                                create_kwargs["compression_opts"] = (
                                    self.compression_opts
                                )
                            if self.shuffle is not None:
                                create_kwargs["shuffle"] = self.shuffle
                        elif self.compression == "zstd":
                            create_kwargs["compression"] = hdf5plugin.Zstd(
                                clevel=self.compression_opts
                            )
                        elif self.compression == "lz4":
                            create_kwargs["compression"] = hdf5plugin.LZ4(nbytes=0)
                        else:
                            raise ValueError(
                                f"Unsupported compression: {self.compression}"
                            )

                        if self.chunk_options is True:
                            create_kwargs["chunks"] = True
                        elif (
                            isinstance(self.chunk_options, tuple)
                            and self.chunk_options is not None
                        ):
                            num_dims = len(data_item.shape)
                            final_chunks = []
                            for i in range(num_dims):
                                if i < len(self.chunk_options):
                                    final_chunks.append(self.chunk_options[i])
                                else:
                                    final_chunks.append(data_item.shape[i])
                            logger.info(f"Final chunks list: {final_chunks}")
                            create_kwargs["chunks"] = tuple(final_chunks)
                        else:
                            logger.info(
                                f"Chunk options: using chunk size {data_item.shape}"
                            )
                            create_kwargs["chunks"] = data_item.shape

                    logger.info(
                        f"Creating dataset for {item_name} with kwargs: {create_kwargs}"
                    )
                    h5file.create_dataset(
                        item_name, data=data_item, **create_kwargs
                    )

                if missing_masks:
                    masks_group = h5file.create_group(
                        self.missing_timesteps_mask_group
                    )
                    for sensor_label, mask_array in missing_masks.items():
                        logger.info(
                            f"Writing missing timesteps mask for {sensor_label} "
                            f"to {h5_path}"
                        )
                        masks_group.create_dataset(sensor_label, data=mask_array)

        return sample_dict

    # ------------------------------------------------------------------
    # Distribution logging
    # ------------------------------------------------------------------

    def _log_sensor_distribution(self, samples: list[TileSample]) -> None:
        """Log per-sensor and combination-level statistics."""
        sensor_counts: dict[str, int] = {}
        combinations: dict[frozenset[str], int] = {}

        for sample in samples:
            for sensor in sample.sensors:
                sensor_counts[sensor.label] = (
                    sensor_counts.get(sensor.label, 0) + 1
                )
            combo = frozenset(s.label for s in sample.sensors)
            combinations[combo] = combinations.get(combo, 0) + 1

        for label, count in sensor_counts.items():
            pct = (count / len(samples)) * 100
            logger.info(f"Sensor {label}: {count} samples ({pct:.1f}%)")

        logger.info("\nSensor combinations:")
        for combo, count in combinations.items():
            pct = (count / len(samples)) * 100
            logger.info(
                f"{'+'.join(sorted(combo))}: {count} samples ({pct:.1f}%)"
            )

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def initialise_h5_dir(self, num_samples: int) -> None:
        """Set the output directory (can only be called once).

        Args:
            num_samples: Number of samples in the dataset.
        """
        if self.h5_dir is not None:
            logger.warning("h5_dir is already set — ignoring new value")
            return

        required_suffix = ""
        if self.required_sensors:
            required_suffix = "_required_" + "_".join(
                sorted(s.label for s in self.required_sensors)
            )
        h5_dir = (
            self.tile_path
            / f"{self.h5_folder}{self.compression_suffix}{self.tile_size_suffix}"
            / (
                "_".join(sorted(s.label for s in self.supported_sensors))
                + required_suffix
            )
            / str(num_samples)
        )
        self.h5_dir = h5_dir
        logger.info(f"Setting h5_dir to {self.h5_dir}")
        os.makedirs(self.h5_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Sample loading helper
    # ------------------------------------------------------------------

    @classmethod
    def _load_and_rearrange(
        cls, tile: SensorTile, sample: TileSample
    ) -> np.ndarray:
        """Load a raster and rearrange to the canonical axis order.

        4-D inputs ``(T, C, H, W)`` become ``(H, W, T, C)``.
        3-D inputs ``(C, H, W)`` become ``(H, W, C)``.
        2-D inputs ``(T, C)`` are returned unchanged.
        """
        image = load_raster_for_sample(tile, sample)

        if image.ndim == 4:
            return rearrange(image, "t c h w -> h w t c")
        elif image.ndim == 3:
            return rearrange(image, "c h w -> h w c")
        elif image.ndim == 2:
            return image
        else:
            raise ValueError(
                f"Unexpected image shape {image.shape} for sensor {tile.sensor.label}"
            )

    # ------------------------------------------------------------------
    # Filtering pipeline
    # ------------------------------------------------------------------

    def _filter_samples(
        self, samples: list[TileSample]
    ) -> list[TileSample]:
        """Apply quality and completeness filters to the sample list."""
        logger.info(f"Number of samples before filtering: {len(samples)}")

        processed = self._process_samples(samples)
        filtered: list[TileSample] = []

        for sample in processed:
            if not all(
                sensor in self.supported_sensors
                for sensor in sample.sensors
                if not sensor.skip_csv_parsing
            ):
                logger.info("Skipping sample with unsupported sensors")
                continue

            if any(
                sensor not in sample.sensors
                for sensor in self.required_sensors
            ):
                logger.info("Skipping sample missing a required sensor")
                continue

            if sample.cadence != TemporalCadence.ANNUAL:
                logger.debug("Skipping non-annual sample")
                continue

            timestamp_dict = sample.extract_timestamps()
            spacetime_sensors = {
                sensor: ts
                for sensor, ts in timestamp_dict.items()
                if sensor.varies_in_space_and_time
            }

            if not spacetime_sensors:
                logger.info("Skipping sample with no spacetime-varying sensors")
                continue

            longest = self._longest_timestamp_array(spacetime_sensors)
            if len(longest) < ANNUAL_STEP_COUNT:
                logger.info(
                    "Skipping sample with fewer than "
                    f"{ANNUAL_STEP_COUNT} timesteps"
                )
                continue

            filtered.append(sample)

        logger.info("Distribution of samples after filtering:")
        self._log_sensor_distribution(filtered)
        return filtered

    def retrieve_and_filter_samples(self) -> list[TileSample]:
        """Parse, assemble, and filter samples in one call."""
        samples = self._retrieve_samples()
        return self._filter_samples(samples)

    # ------------------------------------------------------------------
    # Compression settings
    # ------------------------------------------------------------------

    def save_compression_settings(self) -> None:
        """Persist compression settings as JSON alongside the HDF5 files."""
        if self.h5_dir is None:
            raise ValueError("h5_dir is not set")

        settings = {
            "compression": (
                str(self.compression) if self.compression is not None else None
            ),
            "compression_opts": (
                int(self.compression_opts)
                if self.compression_opts is not None
                else None
            ),
            "shuffle": (
                bool(self.shuffle) if self.shuffle is not None else None
            ),
        }

        path = self.h5_dir / self.compression_settings_fname
        logger.info(f"Saving compression settings to {path}")
        with path.open("w") as f:
            json.dump(settings, f, indent=2)

    # ------------------------------------------------------------------
    # High-level orchestration
    # ------------------------------------------------------------------

    def prepare_h5_dataset(self, samples: list[TileSample]) -> None:
        """Expand samples into sub-tiles, persist metadata, and write HDF5."""
        tuples: list[tuple[int, TileSample]] = []
        for sample in samples:
            for j in range(self.total_subtiles):
                tuples.append((j, sample))

        self.initialise_h5_dir(len(tuples))
        self.save_compression_settings()
        self.save_sample_metadata(tuples)
        self.save_latlon_distribution(tuples)
        logger.info("Creating H5 files — this may take some time…")
        self.create_h5_dataset(tuples)

    def run(self) -> None:
        """Execute the full conversion pipeline."""
        samples = self.retrieve_and_filter_samples()
        self.prepare_h5_dataset(samples)
