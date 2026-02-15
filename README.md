# SpaceNit

SpaceNit is a research toolkit for building **multi-modal, spatiotemporal geospatial foundation models** from scratch -- covering pretraining, evaluation, and downstream adaptation on satellite imagery.

## Capabilities

- **Modeling** -- Transformer-based encoders and decoders designed for heterogeneous remote-sensing tiles with mixed spatial resolutions and temporal cadences.
- **Data pipeline** -- Tile reading, normalization, augmentation, masking/occlusion strategies, and efficient batching for large-scale pretraining.
- **Training** -- Pluggable training runners, objective functions (latent prediction, masked autoencoding, contrastive), distributed training support, and experiment logging.
- **Benchmarks** -- Linear probes, kNN evaluation, fine-tuning harnesses, and adapters for comparing against published baselines.
- **Dataset tooling** -- Utilities for ingesting, converting, and preparing geospatial datasets into the training format.

## Installation

Python 3.12 recommended. We use [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.

```bash
git clone <your-repo-url>
cd spacenit
uv sync --locked --all-groups --python 3.12
```

For inference-only (no training extras):

```bash
uv sync --locked
```

## Quick check

```bash
uv run python -c "import spacenit; print('spacenit loaded')"
```

## Running tests

```bash
# Full suite
uv run --all-groups --no-group flash-attn pytest tests/

# Minimal-dependency model loading tests
uv run --group dev pytest tests_minimal_deps/
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
