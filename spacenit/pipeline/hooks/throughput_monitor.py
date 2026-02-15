"""Throughput monitor hook for the SpaceNit pipeline."""

import logging
import time
from typing import Any

from olmo_core.train.callbacks.speed_monitor import SpeedMonitorCallback

from spacenit.structures import MaskedGeoSample
from spacenit.pipeline.runners.latent_prediction import LatentPredictionRunner
from spacenit.pipeline.runners.contrastive_latent import ContrastiveLatentRunner
from spacenit.pipeline.runners.autoencoder import AutoEncoderRunner
from spacenit.pipeline.runners.dual_branch import DualBranchRunner

logger = logging.getLogger(__name__)


class SpaceNitThroughputMonitor(SpeedMonitorCallback):
    """Throughput monitor hook for the SpaceNit pipeline."""

    priority = 10
    _total_tokens_encoded = 0
    _total_tokens_decoded = 0
    _total_tokens_target_encoder = 0

    def pre_train(self) -> None:
        """Pre-train hook for the throughput monitor."""
        super().pre_train()
        train_module = self.trainer.train_module

        self._token_budget = self.trainer.data_loader.token_budget
        if isinstance(
            train_module,
            AutoEncoderRunner | LatentPredictionRunner | ContrastiveLatentRunner,
        ):
            self._encoder_ratio = getattr(train_module.masking_strategy, "encode_ratio", 0.25)
            self._decoder_ratio = getattr(train_module.masking_strategy, "decode_ratio", 0.25)
            logger.warning(
                "Throughput monitor bases token input on token budget, "
                "encoder ratio, and decoder ratio"
            )
        elif isinstance(train_module, DualBranchRunner):
            self._encoder_ratio = getattr(train_module.masking_strategy_a, "encode_ratio", 0.25)
            self._decoder_ratio = getattr(train_module.masking_strategy_a, "decode_ratio", 0.25)
            enc_b = getattr(train_module.masking_strategy_b, "encode_ratio", 0.25)
            dec_b = getattr(train_module.masking_strategy_b, "decode_ratio", 0.25)
            if enc_b != self._encoder_ratio:
                logger.warning(
                    "Throughput monitor bases token input on encoder ratio "
                    "from masking_strategy_a"
                )
            if dec_b != self._decoder_ratio:
                logger.warning(
                    "Throughput monitor bases token input on decoder ratio "
                    "from masking_strategy_a"
                )
            logger.warning(
                "Throughput monitor bases token input on token budget, "
                "encoder ratio, and decoder ratio"
            )
        else:
            logger.warning(
                "Throughput monitor only calculates token throughput with "
                "AutoEncoderRunner, LatentPredictionRunner, or DualBranchRunner"
            )

    def pre_load_batch(self) -> None:
        """Pre-load batch hook for the throughput monitor."""
        if hasattr(self, "callback_start_time"):
            self.callback_start_time: float
            self.trainer.record_metric(
                "throughput/callback time (s)",
                time.perf_counter() - self.callback_start_time,
            )
        super().pre_load_batch()

    def pre_step(self, batch: Any) -> None:
        """Pre-step hook for the throughput monitor."""
        self._batch_load_time = time.perf_counter() - self._batch_load_start
        if self._first_step:
            return
        # Batch can be 2-tuple (patch_size, sample) or 3-tuple (patch_size, sample_a, sample_b)
        sample: MaskedGeoSample = batch[1]
        self._step_tokens_encoded = (
            sample.num_samples * self._encoder_ratio * self._token_budget
        )
        self._step_tokens_decoded = (
            sample.num_samples * self._decoder_ratio * self._token_budget
        )
        self._step_tokens_target_encoder = sample.num_samples * self._token_budget

        self._total_steps += 1
        self._total_tokens_encoded += self._step_tokens_encoded
        self._total_tokens_decoded += self._step_tokens_decoded
        self._total_tokens_target_encoder += self._step_tokens_target_encoder
        self.model_start_time = time.perf_counter()

    def post_step(self) -> None:
        """Post-step hook for the throughput monitor."""
        counter = time.perf_counter()
        self.model_end_time = counter

        self.trainer.record_metric(
            "throughput/device/data loading (s)", self._batch_load_time
        )
        self._first_step: bool
        if self._first_step:
            self._total_steps = 0
            self._total_tokens = 0
            self._start_time = counter
            self._step_last_logged = counter
            self._first_step = False
            return

        self.model_duration = self.model_end_time - self.model_start_time
        step_time = counter - self._step_last_logged
        total_time = counter - self._start_time
        self._step_last_logged = counter

        bps = 1 / step_time
        bps_avg = self._total_steps / total_time
        self._bps_avg = bps_avg
        data_pct = 100 * self._batch_load_time / step_time
        tps_encoded = self._total_tokens_encoded / step_time
        tps_encoded_avg = self._total_tokens_encoded / total_time
        tps_decoded = self._total_tokens_decoded / step_time
        tps_decoded_avg = self._total_tokens_decoded / total_time
        tps_target_encoder = self._total_tokens_target_encoder / step_time
        tps_target_encoder_avg = self._total_tokens_target_encoder / total_time
        self.trainer.record_metric(
            "throughput/total tokens target encoder-since-restart",
            self._total_tokens_target_encoder,
        )

        self.trainer.record_metric(
            "throughput/total tokens encoded-since-restart", self._total_tokens_encoded
        )
        self.trainer.record_metric(
            "throughput/total tokens decoded-since-restart", self._total_tokens_decoded
        )
        self.trainer.record_metric("throughput/device/TPS Encoded", tps_encoded)
        self.trainer.record_metric(
            "throughput/device/TPS Target Encoder", tps_target_encoder
        )
        self.trainer.record_metric(
            "throughput/device/TPS Target Encoder (estimated avg)",
            tps_target_encoder_avg,
        )
        self.trainer.record_metric(
            "throughput/device/TPS Encoded (estimated avg)", tps_encoded_avg
        )
        self.trainer.record_metric("throughput/device/TPS Decoded", tps_decoded)
        self.trainer.record_metric(
            "throughput/device/TPS Decoded (estimated avg)", tps_decoded_avg
        )
        self.trainer.record_metric("throughput/device/data loading (%)", data_pct)
        self.trainer.record_metric("throughput/device/BPS", bps)
        self.trainer.record_metric("throughput/device/BPS (estimated avg)", bps_avg)
        self.trainer.record_metric(
            "throughput/device/model duration (s)", self.model_duration
        )
        self.trainer.record_metric(
            "throughput/device/model duration (%)", self.model_duration / step_time
        )
        self.callback_start_time = time.perf_counter()
