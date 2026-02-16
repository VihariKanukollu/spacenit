"""End-to-end smoke test for the SpaceNit training and evaluation pipeline.

Tests the full lifecycle with synthetic data:
1. Model construction from config
2. Encoder forward pass (tokenization + transformer)
3. Full LatentPredictor forward (online + target encoder + decoder + projection)
4. Backward pass and gradient flow
5. EMA update
6. Loss computation (contrastive + uniformity)
7. Multiple training steps (simulated training loop)
8. Eval-mode forward pass (feature extraction)
"""

import pytest
import torch
import torch.nn.functional as F

from spacenit.arch.encoder import Decoder, Encoder, EncoderConfig
from spacenit.arch.models import (
    AutoEncoder,
    AutoEncoderConfig,
    DualBranch,
    DualBranchConfig,
    LatentPredictor,
    LatentPredictorConfig,
)
from spacenit.ingestion.sensors import SensorRegistry
from spacenit.pipeline.losses import (
    CompositeLoss,
    ContrastiveLoss,
    LatentPredictionLoss,
    PatchDiscriminationLoss,
    ReconstructionLoss,
    UniformityLoss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")
BATCH = 4
H, W = 16, 16
PATCH = 8
EMBED = 64
HEADS = 4
DEPTH = 2
DECODER_DEPTH = 2
PROJ_DIM = 32


def _sensor_labels(n: int = 1) -> list[str]:
    """Pick the first *n* registered sensor labels."""
    all_labels = SensorRegistry.all_labels()
    candidates = ["sentinel2_l2a", "sentinel1"]
    labels = [l for l in candidates if l in all_labels][:n]
    if len(labels) < n:
        labels = all_labels[:n]
    return labels


def _make_sensor_data(labels: list[str], B: int = BATCH) -> dict[str, torch.Tensor]:
    """Create random sensor tensors in the shape the encoder expects: (B, C, H, W)."""
    data = {}
    for label in labels:
        spec = SensorRegistry.get(label)
        C = spec.total_channels
        data[label] = torch.randn(B, C, H, W, device=DEVICE)
    return data


def _make_encoder_config(labels: list[str]) -> EncoderConfig:
    return EncoderConfig(
        embed_dim=EMBED,
        depth=DEPTH,
        num_heads=HEADS,
        base_patch_size=PATCH,
        sensor_labels=labels,
    )


def _make_latent_config(labels: list[str]) -> LatentPredictorConfig:
    return LatentPredictorConfig(
        encoder=_make_encoder_config(labels),
        decoder_depth=DECODER_DEPTH,
        decoder_num_heads=HEADS,
        ema_momentum=0.99,
        ema_momentum_end=1.0,
        ema_warmup_steps=10,
        projection_dim=PROJ_DIM,
    )


def _make_visibility_mask(
    N_total: int,
    B: int = BATCH,
    encode_ratio: float = 0.5,
) -> torch.Tensor:
    """Create a random visibility mask with VISIBLE_ENCODER (0) and PREDICTED (2).

    ``encode_ratio`` fraction of tokens are visible; the rest are predicted.
    """
    VIS = 0   # TokenVisibility.VISIBLE_ENCODER
    PRED = 2  # TokenVisibility.PREDICTED

    N_vis = max(1, int(N_total * encode_ratio))
    masks = []
    for _ in range(B):
        perm = torch.randperm(N_total)
        m = torch.full((N_total,), PRED, dtype=torch.long)
        m[perm[:N_vis]] = VIS
        masks.append(m)
    return torch.stack(masks)


# ---------------------------------------------------------------------------
# 1. Model construction
# ---------------------------------------------------------------------------


class TestModelConstruction:
    """Verify that all model configs produce valid models."""

    def test_encoder_from_config(self):
        labels = _sensor_labels(1)
        config = _make_encoder_config(labels)
        encoder = config.build()
        assert isinstance(encoder, Encoder)

    def test_latent_predictor_from_config(self):
        labels = _sensor_labels(1)
        config = _make_latent_config(labels)
        model = config.build()
        assert isinstance(model, LatentPredictor)
        assert isinstance(model.encoder, Encoder)
        assert isinstance(model.target_encoder, Encoder)
        assert isinstance(model.decoder, Decoder)

    def test_autoencoder_from_config(self):
        labels = _sensor_labels(1)
        config = AutoEncoderConfig(
            encoder=_make_encoder_config(labels),
            decoder_depth=DECODER_DEPTH,
            decoder_num_heads=HEADS,
            out_channels=SensorRegistry.get(labels[0]).total_channels,
        )
        model = config.build()
        assert isinstance(model, AutoEncoder)

    def test_dual_branch_from_config(self):
        labels = _sensor_labels(1)
        config = DualBranchConfig(
            encoder=_make_encoder_config(labels),
            decoder_depth=DECODER_DEPTH,
            decoder_num_heads=HEADS,
            projection_dim=PROJ_DIM,
            out_channels=SensorRegistry.get(labels[0]).total_channels,
        )
        model = config.build()
        assert isinstance(model, DualBranch)


# ---------------------------------------------------------------------------
# 2. Encoder forward pass
# ---------------------------------------------------------------------------


class TestEncoderForward:
    """Verify encoder tokenization and forward pass shapes."""

    def test_single_sensor_forward(self):
        labels = _sensor_labels(1)
        encoder = _make_encoder_config(labels).build()
        sensor_data = _make_sensor_data(labels)

        encoded, sensor_ids, spatial_ids, temporal_ids, layout = encoder(sensor_data)

        assert encoded.ndim == 3
        assert encoded.shape[0] == BATCH
        assert encoded.shape[2] == EMBED
        assert sensor_ids.shape == encoded.shape[:2]
        assert len(layout) == 1

    def test_multi_sensor_forward(self):
        labels = _sensor_labels(2)
        if len(labels) < 2:
            pytest.skip("Need at least 2 registered sensors")
        encoder = _make_encoder_config(labels).build()
        sensor_data = _make_sensor_data(labels)

        encoded, sensor_ids, spatial_ids, temporal_ids, layout = encoder(sensor_data)

        assert encoded.shape[0] == BATCH
        assert encoded.shape[2] == EMBED
        assert len(layout) == 2

    def test_masked_forward(self):
        labels = _sensor_labels(1)
        encoder = _make_encoder_config(labels).build()
        sensor_data = _make_sensor_data(labels)

        # First do unmasked to get total token count
        full_enc, _, _, _, _ = encoder(sensor_data)
        N_total = full_enc.shape[1]

        # Keep half the tokens
        N_keep = max(1, N_total // 2)
        mask_indices = torch.stack([
            torch.randperm(N_total)[:N_keep] for _ in range(BATCH)
        ])

        masked_enc, masked_ids, _, _, _ = encoder(sensor_data, mask_indices=mask_indices)
        assert masked_enc.shape == (BATCH, N_keep, EMBED)


# ---------------------------------------------------------------------------
# 3. LatentPredictor full forward
# ---------------------------------------------------------------------------


class TestLatentPredictorForward:
    """End-to-end forward pass through the full LatentPredictor."""

    def _forward(self, labels: list[str]):
        config = _make_latent_config(labels)
        model = config.build().to(DEVICE)
        model.eval()

        sensor_data = _make_sensor_data(labels)

        # Get total token count
        with torch.no_grad():
            full_enc, _, _, _, _ = model.encoder(sensor_data)
        N_total = full_enc.shape[1]

        # Build a visibility mask: half visible, half predicted
        vmask = _make_visibility_mask(N_total, B=BATCH, encode_ratio=0.5)
        N_tgt = int((vmask == 2).sum(dim=1)[0].item())

        with torch.no_grad():
            outputs = model(
                sensor_data,
                visibility_mask=vmask,
                patch_size=PATCH,
            )

        return outputs, N_tgt

    def test_output_keys(self):
        labels = _sensor_labels(1)
        outputs, _ = self._forward(labels)
        assert "predictions" in outputs
        assert "targets" in outputs
        assert "online_proj" in outputs
        assert "target_proj" in outputs
        assert "encoder_pooled" in outputs

    def test_output_shapes(self):
        labels = _sensor_labels(1)
        outputs, N_tgt = self._forward(labels)
        assert outputs["predictions"].shape == (BATCH, N_tgt, EMBED)
        assert outputs["targets"].shape == (BATCH, N_tgt, EMBED)
        assert outputs["online_proj"].shape == (BATCH, N_tgt, PROJ_DIM)
        assert outputs["target_proj"].shape == (BATCH, N_tgt, PROJ_DIM)
        assert outputs["encoder_pooled"].shape == (BATCH, EMBED)

    def test_no_mask_forward(self):
        """Forward without a visibility mask (all tokens visible)."""
        labels = _sensor_labels(1)
        config = _make_latent_config(labels)
        model = config.build().to(DEVICE)
        model.eval()
        sensor_data = _make_sensor_data(labels)

        with torch.no_grad():
            outputs = model(sensor_data, patch_size=PATCH)

        assert "predictions" in outputs
        assert "targets" in outputs


# ---------------------------------------------------------------------------
# 4. Backward pass and gradient flow
# ---------------------------------------------------------------------------


class TestBackwardPass:
    """Verify gradients flow through the full model."""

    def test_latent_predictor_gradients(self):
        labels = _sensor_labels(1)
        config = _make_latent_config(labels)
        model = config.build().to(DEVICE)
        model.train()

        sensor_data = _make_sensor_data(labels)

        # Get token count
        with torch.no_grad():
            full_enc, _, _, _, _ = model.encoder(sensor_data)
        N_total = full_enc.shape[1]

        vmask = _make_visibility_mask(N_total, B=BATCH, encode_ratio=0.5)

        outputs = model(
            sensor_data,
            visibility_mask=vmask,
            patch_size=PATCH,
        )

        # Compute a simple loss and backprop
        loss = outputs["predictions"].mean() + outputs["online_proj"].mean()
        loss.backward()

        # Check gradients exist on online encoder params (not target encoder).
        has_any_grad = False
        for name, p in model.encoder.named_parameters():
            if p.grad is not None:
                has_any_grad = True
                assert torch.isfinite(p.grad).all(), f"Non-finite grad for encoder.{name}"
        assert has_any_grad, "No gradients at all in encoder"

        # Target encoder should have no gradients (no_grad)
        for name, p in model.target_encoder.named_parameters():
            assert p.grad is None, f"Unexpected gradient for target_encoder.{name}"


# ---------------------------------------------------------------------------
# 5. EMA update
# ---------------------------------------------------------------------------


class TestEMAUpdate:
    """Verify EMA momentum update works correctly."""

    def test_ema_changes_target(self):
        labels = _sensor_labels(1)
        config = _make_latent_config(labels)
        model = config.build().to(DEVICE)

        # Snapshot target encoder params before
        before = {
            name: p.clone()
            for name, p in model.target_encoder.named_parameters()
        }

        # Perturb online encoder
        with torch.no_grad():
            for p in model.encoder.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Step EMA
        model.step_ema()

        # Target should have changed
        changed = False
        for name, p in model.target_encoder.named_parameters():
            if not torch.allclose(p, before[name]):
                changed = True
                break
        assert changed, "EMA update did not change target encoder"

    def test_ema_momentum_annealing(self):
        labels = _sensor_labels(1)
        config = _make_latent_config(labels)
        model = config.build().to(DEVICE)

        initial_momentum = model._ema_momentum
        for _ in range(5):
            model.step_ema()
        assert model._ema_momentum >= initial_momentum


# ---------------------------------------------------------------------------
# 6. Loss computation
# ---------------------------------------------------------------------------


class TestLosses:
    """Verify all loss functions work with expected tensor shapes."""

    def test_latent_prediction_loss(self):
        loss_fn = LatentPredictionLoss()
        pred = torch.randn(BATCH, 10, EMBED)
        tgt = torch.randn(BATCH, 10, EMBED)
        loss = loss_fn(pred, tgt)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_contrastive_loss(self):
        loss_fn = ContrastiveLoss()
        anchors = F.normalize(torch.randn(BATCH, PROJ_DIM), dim=-1)
        positives = F.normalize(torch.randn(BATCH, PROJ_DIM), dim=-1)
        loss = loss_fn(anchors, positives)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_uniformity_loss(self):
        loss_fn = UniformityLoss()
        embeddings = F.normalize(torch.randn(BATCH, PROJ_DIM), dim=-1)
        loss = loss_fn(embeddings)
        assert loss.ndim == 0

    def test_reconstruction_loss(self):
        loss_fn = ReconstructionLoss(loss_type="mse")
        pred = torch.randn(BATCH, 10, EMBED)
        tgt = torch.randn(BATCH, 10, EMBED)
        loss = loss_fn(pred, tgt)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_composite_loss(self):
        composite = CompositeLoss({
            "latent": (LatentPredictionLoss(), 1.0),
            "contrastive": (ContrastiveLoss(), 0.5),
        })
        loss, individual = composite(
            latent_predictions=torch.randn(BATCH, 10, EMBED),
            latent_targets=torch.randn(BATCH, 10, EMBED),
            contrastive_anchors=F.normalize(torch.randn(BATCH, PROJ_DIM), dim=-1),
            contrastive_positives=F.normalize(torch.randn(BATCH, PROJ_DIM), dim=-1),
        )
        assert loss.ndim == 0
        assert "latent" in individual
        assert "contrastive" in individual


# ---------------------------------------------------------------------------
# 7. Simulated training loop
# ---------------------------------------------------------------------------


class TestTrainingLoop:
    """Simulate multiple training steps end-to-end."""

    def test_multi_step_training(self):
        labels = _sensor_labels(1)
        config = _make_latent_config(labels)
        model = config.build().to(DEVICE)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        contrastive_loss_fn = ContrastiveLoss()
        uniformity_loss_fn = UniformityLoss()

        losses = []

        for step in range(5):
            optimizer.zero_grad()

            sensor_data = _make_sensor_data(labels)

            # Get token count
            with torch.no_grad():
                full_enc, _, _, _, _ = model.encoder(sensor_data)
            N_total = full_enc.shape[1]

            vmask = _make_visibility_mask(N_total, B=BATCH, encode_ratio=0.5)

            outputs = model(
                sensor_data,
                visibility_mask=vmask,
                patch_size=PATCH,
            )

            # Pool projections to (B, D) for contrastive loss
            online_pooled = outputs["online_proj"].mean(dim=1)
            target_pooled = outputs["target_proj"].mean(dim=1)

            # Normalize for contrastive loss
            online_pooled = F.normalize(online_pooled, dim=-1)
            target_pooled = F.normalize(target_pooled, dim=-1)

            # Losses
            c_loss = contrastive_loss_fn(online_pooled, target_pooled)
            u_loss = uniformity_loss_fn(online_pooled)
            total_loss = c_loss + 0.01 * u_loss

            total_loss.backward()

            # Check no NaN gradients
            for name, p in model.named_parameters():
                if p.grad is not None:
                    assert not torch.isnan(p.grad).any(), f"NaN grad at {name}"

            optimizer.step()
            model.step_ema()

            losses.append(total_loss.item())

        # Verify training ran without errors and produced finite losses
        assert all(not (l != l) for l in losses), "NaN loss detected"
        assert len(losses) == 5

    def test_latent_prediction_training(self):
        """Train with latent prediction loss (JEPA-style)."""
        labels = _sensor_labels(1)
        config = _make_latent_config(labels)
        model = config.build().to(DEVICE)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        latent_loss_fn = LatentPredictionLoss()

        for step in range(3):
            optimizer.zero_grad()
            sensor_data = _make_sensor_data(labels)

            with torch.no_grad():
                full_enc, _, _, _, _ = model.encoder(sensor_data)
            N_total = full_enc.shape[1]

            vmask = _make_visibility_mask(N_total, B=BATCH, encode_ratio=0.5)

            outputs = model(
                sensor_data,
                visibility_mask=vmask,
                patch_size=PATCH,
            )

            loss = latent_loss_fn(outputs["predictions"], outputs["targets"])
            loss.backward()
            optimizer.step()
            model.step_ema()

        assert loss.item() > 0


# ---------------------------------------------------------------------------
# 8. Eval-mode forward pass (feature extraction)
# ---------------------------------------------------------------------------


class TestEvalForward:
    """Verify eval-mode feature extraction works."""

    def test_encoder_eval_features(self):
        labels = _sensor_labels(1)
        encoder = _make_encoder_config(labels).build().to(DEVICE)
        encoder.eval()

        sensor_data = _make_sensor_data(labels, B=1)

        with torch.no_grad():
            encoded, sensor_ids, _spatial_ids, _temporal_ids, layout = encoder(sensor_data)

        # Features should be finite
        assert torch.isfinite(encoded).all()
        assert encoded.shape[0] == 1
        assert encoded.shape[2] == EMBED

    def test_latent_predictor_eval_features(self):
        labels = _sensor_labels(1)
        config = _make_latent_config(labels)
        model = config.build().to(DEVICE)
        model.eval()

        sensor_data = _make_sensor_data(labels, B=1)

        with torch.no_grad():
            # Use encoder directly for feature extraction (eval mode)
            encoded, sensor_ids, _spatial_ids, _temporal_ids, layout = model.encoder(sensor_data)

        assert torch.isfinite(encoded).all()

        # Pool features (mean pooling over tokens)
        features = encoded.mean(dim=1)  # (1, D)
        assert features.shape == (1, EMBED)

    def test_autoencoder_eval(self):
        labels = _sensor_labels(1)
        spec = SensorRegistry.get(labels[0])
        config = AutoEncoderConfig(
            encoder=_make_encoder_config(labels),
            decoder_depth=DECODER_DEPTH,
            decoder_num_heads=HEADS,
            out_channels=spec.total_channels,
        )
        model = config.build().to(DEVICE)
        model.eval()

        sensor_data = _make_sensor_data(labels, B=1)

        # PixelHead uses F.fold to reconstruct spatial output, so the
        # number of PREDICTED tokens must equal the spatial patch grid:
        # (H/P) * (W/P).  Build a visibility mask that marks exactly
        # that many tokens as PREDICTED.
        with torch.no_grad():
            full_enc, _, _, _, _ = model.encoder(sensor_data)
        N_total = full_enc.shape[1]
        N_spatial = (H // PATCH) * (W // PATCH)  # 4 for 16x16 with patch=8
        N_vis = N_total - N_spatial

        # First N_vis tokens visible, last N_spatial predicted
        vmask = torch.zeros(1, N_total, dtype=torch.long)
        vmask[:, N_vis:] = 2  # PREDICTED

        with torch.no_grad():
            outputs = model(
                sensor_data,
                visibility_mask=vmask,
                patch_size=PATCH,
                height=H,
                width=W,
            )

        assert "reconstructed" in outputs
        assert "encoded" in outputs
        assert torch.isfinite(outputs["encoded"]).all()


# ---------------------------------------------------------------------------
# 9. Config serialization round-trip
# ---------------------------------------------------------------------------


class TestConfigSerialization:
    """Verify configs can be serialized and deserialized."""

    def test_encoder_config_roundtrip(self):
        labels = _sensor_labels(1)
        config = _make_encoder_config(labels)
        d = config.as_config_dict()
        restored = EncoderConfig.from_dict(d)
        assert restored.embed_dim == config.embed_dim
        assert restored.depth == config.depth
        assert restored.sensor_labels == config.sensor_labels

    def test_latent_predictor_config_roundtrip(self):
        labels = _sensor_labels(1)
        config = _make_latent_config(labels)
        d = config.as_config_dict()
        restored = LatentPredictorConfig.from_dict(d)
        assert restored.decoder_depth == config.decoder_depth
        assert restored.encoder.embed_dim == config.encoder.embed_dim

        # Build from restored config
        model = restored.build()
        assert isinstance(model, LatentPredictor)


# ---------------------------------------------------------------------------
# 10. Multi-sensor training
# ---------------------------------------------------------------------------


class TestMultiSensorTraining:
    """Verify training works with multiple sensors."""

    def test_two_sensor_forward_backward(self):
        labels = _sensor_labels(2)
        if len(labels) < 2:
            pytest.skip("Need at least 2 registered sensors")

        config = _make_latent_config(labels)
        model = config.build().to(DEVICE)
        model.train()

        sensor_data = _make_sensor_data(labels)

        with torch.no_grad():
            full_enc, _, _, _, _ = model.encoder(sensor_data)
        N_total = full_enc.shape[1]

        vmask = _make_visibility_mask(N_total, B=BATCH, encode_ratio=0.5)

        outputs = model(
            sensor_data,
            visibility_mask=vmask,
            patch_size=PATCH,
        )

        loss = outputs["predictions"].mean()
        loss.backward()

        # Verify gradients flow through both sensor patch embedders
        for label in labels:
            embed_module = model.encoder.tokenizer.patch_embeds[label]
            has_grad = any(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in embed_module.parameters()
            )
            assert has_grad, f"No gradient for patch_embeds[{label}]"


# ---------------------------------------------------------------------------
# 11. LatentMIM training loop (patch discrimination + contrastive)
# ---------------------------------------------------------------------------


class TestLatentMIMTraining:
    """Simulate the OLMo-Earth-style training recipe with two views."""

    def test_patch_discrimination_loss(self):
        """PatchDiscriminationLoss produces finite, positive loss."""
        loss_fn = PatchDiscriminationLoss(tau=0.1)
        pred = torch.randn(BATCH, 10, EMBED)
        tgt = torch.randn(BATCH, 10, EMBED)
        loss = loss_fn(pred, tgt)
        assert loss.ndim == 0
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_patch_discrimination_empty(self):
        """PatchDiscriminationLoss handles zero-token case."""
        loss_fn = PatchDiscriminationLoss(tau=0.1)
        pred = torch.randn(BATCH, 0, EMBED)
        tgt = torch.randn(BATCH, 0, EMBED)
        loss = loss_fn(pred, tgt)
        assert loss.ndim == 0

    def test_two_view_mim_training(self):
        """Full two-view MIM training loop: patch disc + InfoNCE on encoder pooled."""
        labels = _sensor_labels(1)
        config = _make_latent_config(labels)
        model = config.build().to(DEVICE)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        patch_disc_fn = PatchDiscriminationLoss(tau=0.1)
        contrastive_fn = ContrastiveLoss(initial_temperature=0.1)

        losses = []
        for step in range(3):
            optimizer.zero_grad()

            sensor_data = _make_sensor_data(labels)

            # Get token count
            with torch.no_grad():
                full_enc, _, _, _, _ = model.encoder(sensor_data)
            N_total = full_enc.shape[1]

            # Two independent visibility masks (simulating two views)
            vmask_a = _make_visibility_mask(N_total, B=BATCH, encode_ratio=0.5)
            vmask_b = _make_visibility_mask(N_total, B=BATCH, encode_ratio=0.5)

            # Forward view A
            out_a = model(
                sensor_data,
                visibility_mask=vmask_a,
                patch_size=PATCH,
            )

            # Forward view B
            out_b = model(
                sensor_data,
                visibility_mask=vmask_b,
                patch_size=PATCH,
            )

            # Patch discrimination on raw decoder output vs target encoder
            mim_a = patch_disc_fn(out_a["predictions"], out_a["targets"])
            mim_b = patch_disc_fn(out_b["predictions"], out_b["targets"])
            mim_loss = 0.5 * (mim_a + mim_b)

            # InfoNCE between mean-pooled encoder outputs of the two views
            pooled_a = F.normalize(out_a["encoder_pooled"], dim=-1)
            pooled_b = F.normalize(out_b["encoder_pooled"], dim=-1)
            con_loss = contrastive_fn(pooled_a, pooled_b)

            total_loss = mim_loss + 0.1 * con_loss
            total_loss.backward()

            # Check no NaN gradients
            for name, p in model.named_parameters():
                if p.grad is not None:
                    assert not torch.isnan(p.grad).any(), f"NaN grad at {name}"

            optimizer.step()
            model.step_ema()
            losses.append(total_loss.item())

        assert all(not (l != l) for l in losses), "NaN loss detected"
        assert len(losses) == 3

        # Verify decoder got gradients (it's no longer frozen)
        has_decoder_grad = any(
            p.grad is not None for p in model.decoder.parameters()
        )
        assert has_decoder_grad, "Decoder should receive gradients in MIM training"
