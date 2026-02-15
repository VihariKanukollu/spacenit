# SpaceNit Evaluation Guide

This guide explains how we launch evaluations for SpaceNit checkpoints and baseline models, including KNN, linear probing, and finetuning jobs.

---

## Table of Contents

1. [Evaluation Overview](#evaluation-overview)
2. [Datasets & Model Checkpoints](#datasets--model-checkpoints)
3. [Quick Start](#quick-start)
4. [KNN / Linear Probing](#knn--linear-probing)
5. [Finetune Sweep](#finetune-sweep)
6. [Monitoring & Outputs](#monitoring--outputs)
7. [Helpful Files](#helpful-files)

---

## Evaluation Overview

We run evaluations through the same `spacenit/ops/experiment.py` entrypoint used for pretraining. The helper scripts below build the underlying launch commands:

- `spacenit/ops/eval_sweep.py` runs KNN (classification) and linear probing (segmentation) sweeps for SpaceNit checkpoints or baseline models, with optional sweeps over learning rate, pretrained / dataset normalizers, and pooling (mean or max).
- `spacenit/ops/eval_sweep_finetune.py` runs fine-tuning sweeps for SpaceNit checkpoints or baseline models, with optional sweeps over learning rate and pretrained / dataset normalizers.

Both scripts use:
- [`spacenit/ops/run_all_evals.py`](../spacenit/ops/run_all_evals.py) for the task registry (`EVAL_TASKS` for KNN and linear probing, and `FT_EVAL_TASKS` for fine-tuning).
- [`spacenit/benchmarks`](../spacenit/benchmarks) for dataset and model wrappers.

Every launch uses one of the evaluation subcommands in `experiment.py`:
- `dry_run_evaluate` prints the config (no execution) for quick checks.
- `evaluate` runs the job locally.
- `launch_evaluate` submits the job to a cluster.

### Prerequisites

- Python environment configured as described in [Pretraining.md](Pretraining.md#environment-setup).
- One 80 GB GPU (A100 or H100 recommended). If you see OOM errors when running some tasks, consider reducing the batch size.

### Supported Models

- **SpaceNit models:** Nano, Tiny, Base, and Large size.
- **Others:** Supported baseline models are defined in `spacenit/benchmarks/adapters/__init__.py`, which includes Galileo, Satlas, Prithvi v2, Panopticon, CROMA, AnySat etc. Multi-size variants (if available) are also supported.

---

## Datasets & Model Checkpoints

- **Evaluation datasets**
  - Follow the download instructions in [Pretraining.md](Pretraining.md#evaluation-datasets).
- **SpaceNit checkpoints**
  - Clone the release repos from Hugging Face, e.g.:
    ```bash
    git clone git@hf.co:spacenit/SpaceNit-v1-Nano
    git clone git@hf.co:spacenit/SpaceNit-v1-Tiny
    git clone git@hf.co:spacenit/SpaceNit-v1-Base
    git clone git@hf.co:spacenit/SpaceNit-v1-Large
    ```
  - Pass the desired checkpoint directory via `--checkpoint_path` and the corresponding `--module_path` when running the evaluation sweeps.

- **Baselines**: When using `--model=<name>`, some models (e.g., AnySat, Panopticon) will automatically download checkpoints from Hugging Face or TorchHub. Other models require manually downloading their checkpoints and setting the model path in the config.

---

## Quick Start

### 1. Activate your environment

```bash
source .venv/bin/activate
```

### 2. Run a dry run to inspect the commands

```bash
python -m spacenit.ops.eval_sweep \
  --cluster=local \
  --checkpoint_path=/your/path/to/SpaceNit-v1-Base \
  --module_path=scripts/official/base.py \
  --defaults_only \
  --dry_run
```

This prints the exact command without executing it.

### 3. Launch for real

Remove `--dry_run` once the command looks correct. Pick the launch target you need:

- **Local GPUs (`--cluster=local`)**

  ```bash
  python -m spacenit.ops.eval_sweep \
    --cluster=local \
    --checkpoint_path=/your/path/to/SpaceNit-v1-Base \
    --module_path=scripts/official/base.py \
    --project_name=spacenit_evals \
    --defaults_only
  ```

---

## KNN / Linear Probing

Use `spacenit/ops/eval_sweep.py` for KNN and linear probing tasks.

### Required flags

- `--cluster`: Cluster identifier (set to `local` for local GPUs).
- `--module_path`: Override the launch module (defaults to the model-specific launcher).
- Exactly one of:
  - `--checkpoint_path`: Passing SpaceNit checkpoint.
  - `--model=<baseline_name>` or `--model=all`: Evaluate baseline models defined in [`benchmarks/adapters`](../spacenit/benchmarks/adapters).

### Common optional flags

- `--project_name`: W&B project (defaults to `EVAL_WANDB_PROJECT`).
- `--defaults_only`: Run a single command using the default lr / normalization / pooling.
- `--lr_only`: Sweep learning rates but keep normalization + pooling at defaults.
- `--all_sizes` or `--size=<variant>`: Evaluate every published size for multi-size baselines.
- `--model-skip-names=a,b`: Skip a subset when using `--model=all`.
- `--select_best_val`: Uses validation metric to pick the best epoch before reporting test metrics.
- `--dry_run`: Print commands without launching.

### Example: Local run for SpaceNit
```bash
python -m spacenit.ops.eval_sweep \
  --cluster=local \
  --checkpoint_path=/your/path/to/SpaceNit-v1-Nano \
  --module_path=scripts/official/nano.py \
  --project_name=spacenit_evals \
  --select_best_val \
  --trainer.callbacks.downstream_evaluator.run_on_test=True \
  --trainer.callbacks.downstream_evaluator.tasks_to_run=\[m_eurosat\] \
  --defaults_only
```

### Example: Local run for Galileo
```bash
python -m spacenit.ops.eval_sweep \
  --cluster=local \
  --model=galileo \
  --all_sizes \
  --select_best_val \
  --project_name=baselines_evals \
  --trainer.callbacks.downstream_evaluator.run_on_test=True \
  --trainer.callbacks.downstream_evaluator.tasks_to_run=\[m_eurosat\] \
  --defaults_only
```

---

## Finetune Sweep

Use `spacenit/ops/eval_sweep_finetune.py` for fine-tuning tasks.

### Required flags

- `--cluster`: Cluster identifier (set to `local` for local GPUs).
- One of:
  - `--checkpoint_path`: Fine-tune a SpaceNit checkpoint.
  - `--model=<preset_key>`: Use a baseline preset (choices listed in the script's help).

### Fine-tune specific flags

- `--defaults_only`: Run only the first learning rate in `FT_LRS`.
- `--module_path`: Override the launch script (defaults to the preset's launcher).
- `--use_dataset_normalizer`: Force dataset statistics even when a preset has its own pretrained normalizer.
- `--finetune_seed`: Set a random base seed for running the downstream tasks.
- `--dry_run`: Print commands without launching.

### Example: Local run for SpaceNit
```bash
python -m spacenit.ops.eval_sweep_finetune \
  --cluster=local \
  --checkpoint_path=/your/path/to/SpaceNit-v1-Base \
  --module_path=scripts/official/base.py \
  --project_name=spacenit_evals \
  --defaults_only
```

### Example: Local run for a baseline using dataset normalizer
```bash
python -m spacenit.ops.eval_sweep_finetune \
  --cluster=local \
  --model=terramind \
  --project_name=baseline_evals \
  --use_dataset_normalizer \
  --defaults_only
```

---

## Monitoring & Outputs

- **W&B logging:** Both scripts default to `EVAL_WANDB_PROJECT`. Override with `--project_name` or disable W&B via `--trainer.callbacks.wandb.enabled=False`.
- **Inspecting results:** Use [`scripts/tools/get_max_eval_metrics_from_wandb.py`](../scripts/tools/get_max_eval_metrics_from_wandb.py) to pull the best metric per task across runs.

---

## Helpful Files

- [`benchmarks/adapters`](../spacenit/benchmarks/adapters): Baseline models and their launchers.
- [`benchmarks/benchmark_adapter.py`](../spacenit/benchmarks/benchmark_adapter.py): Eval wrapper contract to be able to run evals on various models.
- [`benchmarks/datasets`](../spacenit/benchmarks/datasets/): Dataset loaders and shared dataset utils.
- [`benchmarks/datasets/constants.py`](../spacenit/benchmarks/datasets/constants.py): Dataset definitions (paths, splits, normalization) used to build commands.
