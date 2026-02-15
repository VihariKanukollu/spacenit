# SpaceNit Pretraining Guide

This guide walks you through setting up and running pretraining jobs for SpaceNit.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Launching Scripts](#launching-scripts)
3. [Dataset Setup](#dataset-setup)
4. [Experiment Tracking](#experiment-tracking)
5. [Official Training Scripts](#official-training-scripts)
6. [Ablations](#ablations)
7. [Overrides and Experiments](#overrides-and-experiments)
8. [Helpful Files for Understanding](#helpful-files-for-understanding)

---

## Environment Setup

### Python Environment Setup

**Prerequisites:** Python 3.11 or higher (Python 3.12 recommended)

Follow the setup instructions in the [main README](../README.md#general-setup) to install dependencies using `uv`.

Once setup is complete, activate the virtual environment before running training scripts:
```bash
source .venv/bin/activate
```

### Running on Docker

We run our training scripts using a Docker image with PyTorch and CUDA support.

**Important Notes:**
- The code from this repository is **not included** in the Docker image to aid in active development. The code is mounted or copied at runtime.
- You may need to adapt the Docker image for your hardware and CUDA version.

## Launching Scripts

### Basic Command Structure

All training scripts configure builder functions used in the main entrypoint: [`main` in `spacenit/ops/experiment.py`](../spacenit/ops/experiment.py).

**General Usage:**
```bash
python scripts/official/<SCRIPT>.py <SUBCOMMAND> <RUN_NAME> <CLUSTER> [OVERRIDES...]
```

For multi-gpu training, use `torchrun`:
```bash
torchrun [TORCHRUN_OPTIONS] scripts/official/<SCRIPT>.py <SUBCOMMAND> <RUN_NAME> <CLUSTER> [OVERRIDES...]
```

#### Available Subcommands

The most commonly used subcommands are:
- **`train`**: Run distributed training (use with `torchrun` for multi-GPU)
- **`launch`**: Submit a cluster job
- **`dry_run`**: Validate configuration without running training

#### Command Structure Examples

**Example 1: Single-GPU Training for Debugging**
```bash
torchrun scripts/official/nano.py train my_debug_run local \
  --dataset.h5py_dir=/path/to/data \
  --data_loader.global_batch_size=64
```

**Example 2: Multi-GPU Training with torchrun**
```bash
torchrun --nproc_per_node=4 scripts/official/base.py train my_pretrain_run local \
  --dataset.h5py_dir=/path/to/data \
  --data_loader.global_batch_size=512
```

**Example 3: Validate Configuration Without Training**
```bash
python scripts/official/base.py dry_run my_config_test local \
  --trainer.max_duration.value=100
```

## Dataset Setup

### Training Dataset Setup

The pretraining dataset can be downloaded from Hugging Face:

1. Download dataset:
   ```bash
   hf download spacenit/spacenit_pretrain_dataset --repo-type dataset --local-dir /path/to/save/location --include "h5py_data/*"
   ```
2. Extract dataset from tar files:
   ```bash
   export H5_DIR=/path/to/extraction/location/
   export TAR_DIR=/path/to/save/location/

   mkdir -p "$H5_DIR"
   cd "$TAR_DIR"

   # run one tar per core; tune -P if disk gets saturated
   find . -maxdepth 1 -name '*.tar' -print0 \
   | xargs -0 -n1 -P"$(nproc)" -I{} \
     tar -xf "{}" -C "$H5_DIR" \
       --no-same-owner --numeric-owner --delay-directory-restore
   ```

### Dataset Path Configuration

You must specify the dataset path when launching training scripts:

```bash
--dataset.h5py_dir=/your/path/to/h5data/num_samples
```

### Evaluation Datasets

Evaluation datasets have default paths set in [`spacenit/benchmarks/datasets/constants.py`](../spacenit/benchmarks/datasets/constants.py).

You may need to download evaluation datasets and set environment variables:

```bash
export GEOBENCH_DIR="/your/path/to/research_benchmarks/geobench"
export MADOS_DIR="/your/path/to/research_benchmarks/mados"
export FLOODS_DIR="/your/path/to/research_benchmarks/floods"
export PASTIS_DIR="/your/path/to/research_benchmarks/pastis_r"
```

If you wish to only use a subset of the evaluations, add the following override:
```bash
--trainer.callbacks.downstream_evaluator.tasks_to_run=\[mados,pastis_sentinel2\]
```

If you do not want to run **any** evaluations during training:
```bash
--trainer.callbacks.downstream_evaluator.enabled=False
```

---

## Experiment Tracking

#### W&B API Key (For Logging)

```bash
export WANDB_API_KEY="your_wandb_api_key_here"
```

Alternatively, you can disable W&B logging in your configuration:
```bash
--trainer.callbacks.wandb.enabled=False
```

## Official Training Scripts

All official release scripts can be found at [`scripts/official/`](../scripts/official/).
Below is a table demonstrating how to launch various model sizes using `torchrun`. Adjust the dataset path and configuration overrides as needed for your setup.

| Model Size | Script | Hardware | Example Command |
|------------|--------|----------|-----------------|
| **Nano** | `scripts/official/nano.py` | 4x GPUs (16GB+ VRAM each) | `torchrun --nproc_per_node=4 scripts/official/nano.py train nano_run local` |
| **Tiny** | `scripts/official/tiny.py` | 4-8x GPUs (24GB+ VRAM each) | `torchrun --nproc_per_node=4 scripts/official/tiny.py train tiny_run local` |
| **Base** | `scripts/official/base.py` | 8x GPUs (40GB+ VRAM each) | `torchrun --nproc_per_node=8 scripts/official/base.py train base_run local` |
| **Large** | `scripts/official/large.py` | 8x GPUs (80GB VRAM each) | `torchrun --nproc_per_node=8 scripts/official/large.py train large_run local` |

> **Hardware Adaptation Note:**
> You may need to adapt parameters depending on your available hardware:
> - **Limited VRAM:** Reduce `--data_loader.global_batch_size` and/or `--train_module.rank_microbatch_size`
> - **Fewer GPUs:** Adjust `--nproc_per_node` and scale batch size accordingly
> - **Different GPU types:** Monitor memory usage and adjust batch sizes to avoid OOM errors
> - **CPU constraints:** Adjust `--data_loader.num_workers` based on available CPU cores

**Specifying Dataset Path:**

You must specify the path to your HDF5 dataset directory by adding `--dataset.h5py_dir=/path/to/h5py_dir` to any command above.

**Example:**
```bash
torchrun --nproc_per_node=4 scripts/official/nano.py train nano_run local \
  --dataset.h5py_dir=/path/to/h5py_dir
```

**Example with Base model:**
```bash
torchrun --nproc_per_node=8 scripts/official/base.py train base_run local \
  --dataset.h5py_dir=/path/to/your/h5data/num_samples
```

> **Checkpoint Saving Note:**
> When using `local` as the cluster argument, checkpoints are automatically saved to `./local_output`. You can override this location with `--common.save_folder=path/to/savefolder`.

## Ablations

Ablation studies isolate the impact of specific components in the base model configuration. All ablations can be launched similarly to the official training scripts.

### Available Ablations

#### Loss & Training Strategy Ablations

- **No contrastive loss** - Disables the contrastive loss component (sets weight to 0.0)
- **Random masking** - Uses random masking instead of structured masking strategy
- **MAE (Masked Autoencoder)** - Switches to a pure MAE training approach
- **Random target init** - Reinitializes target projections randomly instead of using pretrained weights
- **Original patch disc loss** - Uses the legacy patch discrimination loss implementation
- **EMA active** - Re-enables exponential moving average for target encoder

#### Modality Ablations

- **No ag maps** - Removes agricultural map modalities (WorldCereal, CDL)
- **No maps** - Removes all map modalities (ag maps + WorldCover, OpenStreetMap, canopy height)
- **No decode modalities** - Removes all decode-only modalities (maps + SRTM)
- **No Landsat** - Removes Landsat imagery
- **S2 only** - Sentinel-2 only (removes Sentinel-1 as well)

### Running Ablations

Run specific ablations with `torchrun`:

Example - No contrastive loss ablation:
```bash
torchrun --nproc_per_node=8 scripts/official/base.py train base_no_contrastive local \
  --train_module.contrastive_config.loss_config.weight=0.0 \
  --dataset.h5py_dir=/path/to/data
```

Example - S2-only modality ablation:
```bash
torchrun --nproc_per_node=8 scripts/official/base.py train base_s2_only local \
  --common.training_modalities='[sentinel2_l2a]' \
  --train_module.masking_config.strategy_config.only_decode_modalities='[]' \
  --dataset.h5py_dir=/path/to/data
```

---

## Overrides and Experiments

### How Overrides Work

The experiment framework uses a builder pattern with override capabilities. Launch scripts can be edited to change the configuration or you can override any configuration parameter via CLI arguments using dotted notation.

### Example: Custom Training Run with Multiple Overrides

```bash
torchrun --nproc_per_node=8 scripts/official/base.py train custom_experiment local \
  --data_loader.global_batch_size=256 \
  --data_loader.num_workers=8 \
  --train_module.rank_microbatch_size=8 \
  --train_module.optim_config.lr=0.0002 \
  --train_module.scheduler.warmup_steps=5000 \
  --trainer.max_duration.epochs=100
  # Optionally --dataset.h5py_dir=/your/path/to/data \
```

## Dataset Directory and File Structure

The H5 dataset follows a hierarchical directory structure:

```
<tile_path>/
  h5py_data_w_missing_timesteps[_compression_settings][_tilesize_x_numsubtiles]/
    <sorted_sensor_names>[_required_<required_sensors>]/
      <num_samples>/
        sample_0.h5
        sample_1.h5
        ...
        sample_metadata.csv
        latlon_distribution.npy
        compression_settings.json
```

#### Core Files in Each Dataset

1. **`sample_{index}.h5`** - Individual sample files containing:
   - **`latlon`**: Float32 array `[lat, lon]` - geographic coordinates
   - **`timestamps`**: Integer array `[T, 3]` where T=time steps, columns are `[day, month, year]`
   - **Sensor datasets**: Named by sensor (e.g., `"sentinel2"`, `"era5_10"`, `"naip"`, `"landsat"` - see all available sensors in [`modalities.py`](../spacenit/ingestion/modalities.py))
     - Spatial sensors: Shape `[H, W, T, C]` or `[H, W, C]` depending on temporal variation
     - Non-spatial sensors: Shape `[T, C]`
   - **`missing_timesteps_masks/`** group: Boolean masks per sensor (shape `[T]`) indicating which timestamps from the longest timestamp array are present for that specific sensor

2. **`sample_metadata.csv`** - CSV with columns `sample_index, <sensor1>, <sensor2>...` where values are 1 (present) or 0 (absent), tracking which sensors exist in each sample

3. **`latlon_distribution.npy`** - NumPy array `[N, 2]` of all sample lat/lons for dataset statistics

4. **`compression_settings.json`** - Stores compression algorithm, compression level options, and shuffle filter settings used for all H5 files

**Key Invariant:** All H5 files follow the same schema with `latlon`, `timestamps`, sensor datasets, and `missing_timesteps_masks` group structure, ensuring consistency across the entire dataset.


## Helpful Files for Understanding

### Configuration Files
- [`scripts/official/base.py`](../scripts/official/base.py) - Main entry point, model config
- [`scripts/official/script.py`](../scripts/official/script.py) - All component builders (dataset, dataloader, trainer, callbacks)
- [`spacenit/benchmarks/datasets/constants.py`](../spacenit/benchmarks/datasets/constants.py) - Evaluation dataset path configuration

### Dataset Files
- [`spacenit/ingestion/tile_dataset.py`](../spacenit/ingestion/tile_dataset.py) - Dataset implementation and configuration
- [`spacenit/ingestion/batch_assembly.py`](../spacenit/ingestion/batch_assembly.py) - Dataloader implementation and configuration
- [`spacenit/ingestion/modalities.py`](../spacenit/ingestion/modalities.py) - Sensor definitions and constants

### Training Files
- [`spacenit/ops/experiment.py`](../spacenit/ops/experiment.py) - Core experiment orchestration
- [`spacenit/pipeline/runners/`](../spacenit/pipeline/runners/) - Training runner implementations
- [`spacenit/pipeline/occlusion.py`](../spacenit/pipeline/occlusion.py) - Occlusion (masking) policy implementations

---
