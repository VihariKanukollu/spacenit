"""SpaceNit Base model training configuration."""

import logging
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from script import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
    MAX_PATCH_SIZE,
)

from spacenit.arch.encoder import EncoderConfig
from spacenit.arch.models import LatentPredictorConfig
from spacenit.ops.experiment import CommonComponents, main
from spacenit.ops.helpers import MODEL_SIZE_ARGS

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> LatentPredictorConfig:
    """Build the model config for a Base experiment."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]

    return LatentPredictorConfig(
        encoder=EncoderConfig(
            embed_dim=model_size["encoder_embedding_size"],
            num_heads=model_size["encoder_num_heads"],
            depth=model_size["encoder_depth"],
            ffn_expansion=model_size["mlp_ratio"],
            drop_path=0.1,
            # Must match the dataloader's max_patch_size (and OLMo-Earth).
            base_patch_size=MAX_PATCH_SIZE,
            sensor_labels=common.training_modalities,
            tokenization_config=common.tokenization_config,
        ),
        decoder_depth=model_size["decoder_depth"],
        decoder_num_heads=model_size["decoder_num_heads"],
        # Frozen target encoder (matching OLMo-Earth's ema_decay=(1.0, 1.0)).
        ema_momentum=1.0,
        ema_momentum_end=1.0,
        # Match OLMo-Earth: contrastive projection dim equals encoder dim.
        projection_dim=model_size["encoder_embedding_size"],
    )


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
