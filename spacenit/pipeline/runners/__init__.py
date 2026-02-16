"""SpaceNit training runners."""

from spacenit.pipeline.runners.base_runner import (
    SpaceNitTrainRunner,
    SpaceNitTrainRunnerConfig,
)
from spacenit.pipeline.runners.latent_prediction import (
    LatentPredictionRunner,
    LatentPredictionRunnerConfig,
)
from spacenit.pipeline.runners.autoencoder import (
    AutoEncoderRunner,
    AutoEncoderRunnerConfig,
)
from spacenit.pipeline.runners.contrastive_latent import (
    ContrastiveLatentRunner,
    ContrastiveLatentRunnerConfig,
)
from spacenit.pipeline.runners.contrastive_latent_mim import (
    ContrastiveLatentMIMRunner,
    ContrastiveLatentMIMRunnerConfig,
)
from spacenit.pipeline.runners.dual_branch import (
    DualBranchRunner,
    DualBranchRunnerConfig,
)

__all__ = [
    "SpaceNitTrainRunner",
    "SpaceNitTrainRunnerConfig",
    "LatentPredictionRunner",
    "LatentPredictionRunnerConfig",
    "AutoEncoderRunner",
    "AutoEncoderRunnerConfig",
    "ContrastiveLatentRunner",
    "ContrastiveLatentRunnerConfig",
    "ContrastiveLatentMIMRunner",
    "ContrastiveLatentMIMRunnerConfig",
    "DualBranchRunner",
    "DualBranchRunnerConfig",
]
