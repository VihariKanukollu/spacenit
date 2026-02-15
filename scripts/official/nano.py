"""SpaceNit Nano model training configuration."""

import logging
import sys
from pathlib import Path

# Ensure the script directory is on sys.path so sibling modules resolve
# regardless of the working directory (e.g. when invoked via torchrun from
# the repository root).
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from script import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
)

from spacenit.arch.encoder import EncoderConfig
from spacenit.arch.models import LatentPredictorConfig
from spacenit.ops.experiment import CommonComponents, main
from spacenit.ops.helpers import MODEL_SIZE_ARGS

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> LatentPredictorConfig:
    """Build the model config for a Nano experiment."""
    model_size = MODEL_SIZE_ARGS["nano"]

    return LatentPredictorConfig(
        encoder=EncoderConfig(
            embed_dim=model_size["encoder_embedding_size"],
            num_heads=model_size["encoder_num_heads"],
            depth=model_size["encoder_depth"],
            ffn_expansion=model_size["mlp_ratio"],
            sensor_labels=common.training_modalities,
        ),
        decoder_depth=model_size["decoder_depth"],
        decoder_num_heads=model_size["decoder_num_heads"],
    )


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
    )
