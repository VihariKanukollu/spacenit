"""Load SpaceNit pre-trained models from Hugging Face.

This module works with or without olmo-core installed:
- Without olmo-core: inference-only mode (loading pre-trained models)
- With olmo-core: full functionality including training

The weights are converted to pth file from distributed checkpoint like this:

    import json
    from pathlib import Path

    import torch

    from olmo_core.config import Config
    from olmo_core.distributed.checkpoint import load_model_and_optim_state

    checkpoint_path = Path("/path/to/checkpoints/nano_lr0.001_wd0.002/step370000")
    with (checkpoint_path / "config.json").open() as f:
        config_dict = json.load(f)
        model_config = Config.from_dict(config_dict["model"])

    model = model_config.build()

    train_module_dir = checkpoint_path / "model_and_optim"
    load_model_and_optim_state(str(train_module_dir), model)
    torch.save(model.state_dict(), "SpaceNit-v1-Nano.pth")
"""

import json
from enum import StrEnum
from os import PathLike

import torch
from huggingface_hub import hf_hub_download
from upath import UPath

from spacenit.settings import Config

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "weights.pth"


class PretrainedModelID(StrEnum):
    """SpaceNit pre-trained model ID."""

    SPACENIT_V1_NANO = "SpaceNit-v1-Nano"
    SPACENIT_V1_TINY = "SpaceNit-v1-Tiny"
    SPACENIT_V1_BASE = "SpaceNit-v1-Base"
    SPACENIT_V1_LARGE = "SpaceNit-v1-Large"

    def hf_repo(self) -> str:
        """Return the Hugging Face repo ID for this model."""
        return f"spacenit/{self.value}"


def load_pretrained(model_id: PretrainedModelID, load_weights: bool = True) -> torch.nn.Module:
    """Initialize and load the weights for the specified model from Hugging Face.

    Args:
        model_id: the model ID to load.
        load_weights: whether to load the weights. Set false to skip downloading the
            weights from Hugging Face and leave them randomly initialized. Note that
            the config.json will still be downloaded from Hugging Face.
    """
    config_fpath = _locate_artifact(model_id, CONFIG_FILE)
    model = _build_from_config(config_fpath)

    if not load_weights:
        return model

    state_dict_fpath = _locate_artifact(model_id, WEIGHTS_FILE)
    state_dict = _read_state_dict(state_dict_fpath)
    model.load_state_dict(state_dict)
    return model


def load_from_directory(
    model_path: PathLike | str, load_weights: bool = True
) -> torch.nn.Module:
    """Initialize and load the weights for the specified model from a path.

    Args:
        model_path: the path to the model.
        load_weights: whether to load the weights. Set false to skip loading the
            weights and leave them randomly initialized.
    """
    config_fpath = _locate_artifact(model_path, CONFIG_FILE)
    model = _build_from_config(config_fpath)

    if not load_weights:
        return model

    state_dict_fpath = _locate_artifact(model_path, WEIGHTS_FILE)
    state_dict = _read_state_dict(state_dict_fpath)
    model.load_state_dict(state_dict)
    return model


def _locate_artifact(
    model_id_or_path: PretrainedModelID | PathLike | str, filename: str
) -> UPath:
    """Resolve the artifact file path for the specified model ID or path, downloading it from Hugging Face if necessary."""
    if isinstance(model_id_or_path, PretrainedModelID):
        return UPath(
            hf_hub_download(repo_id=model_id_or_path.hf_repo(), filename=filename)  # nosec
        )
    base = UPath(model_id_or_path)
    return base / filename


def _build_from_config(path: UPath) -> torch.nn.Module:
    """Build the model from the config at the specified path."""
    with path.open() as f:
        config_dict = json.load(f)
        model_config = Config.from_dict(config_dict["model"])
    return model_config.build()


def _read_state_dict(path: UPath) -> dict[str, torch.Tensor]:
    """Load the model state dict from the specified path."""
    with path.open("rb") as f:
        state_dict = torch.load(f, map_location="cpu")
    return state_dict
