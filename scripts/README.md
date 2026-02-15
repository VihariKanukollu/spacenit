# Scripts

This directory contains training and utility scripts for SpaceNit.

## Structure

- **`official/`** — Official training configurations for each model size (Nano, Tiny, Base, Large). Each size-specific script defines the model architecture and delegates shared training logic to `script.py`.

## Usage

Training scripts are launched via `torchrun` or a cluster scheduler. For example:

```bash
torchrun --nproc_per_node=8 scripts/official/nano.py --run_name my_run --cluster local
```

See individual scripts for model-specific details.
