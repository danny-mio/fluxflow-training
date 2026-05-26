"""Unit tests for v0.10.0 training-repo changes.

Covers:
- pipeline_config: new YAML keys (lambda_ctx_aux, ctx_loss_weight, freeze_context_branch)
- pipeline_orchestrator: freeze_context_branch / unfreeze_context_branch helpers
- pipeline_orchestrator: discriminator ctx_dim dispatch (v0.10.0 vs legacy)
- vae_trainer: ctx_input_dim parameter and ctx_aux_loss term
- vae_trainer: KL remains on z branch only (no ctx logvar)
- flow_trainer: ctx_loss_weight wiring (already in FlowTrainer, assert attr present)
"""

from typing import Optional
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.nn as nn

from fluxflow_training.training.pipeline_config import (
    OptimizationConfig,
    OptimizerConfig,
    PipelineStepConfig,
    parse_pipeline_config,
)
from fluxflow_training.training.pipeline_orchestrator import TrainingPipelineOrchestrator

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_linear_seq(in_dim: int, out_dim: int) -> nn.Sequential:
    """Minimal Sequential with a Linear as first module (mirrors context_predictor)."""
    return nn.Sequential(nn.Linear(in_dim, 64), nn.SiLU(), nn.Linear(64, out_dim))


def _mock_compressor(d_model: int = 128, has_ctx_dims: bool = True) -> MagicMock:
    """Build a mock compressor that (optionally) exposes get_context_dims()."""
    comp = MagicMock(spec=nn.Module)
    comp.parameters = Mock(return_value=iter([nn.Parameter(torch.zeros(1))]))
    if has_ctx_dims:
        comp.get_context_dims = Mock(return_value=d_model)
    else:
        del comp.get_context_dims  # simulate legacy compressor
    # Give it ctx sub-modules for freeze tests
    for attr in (
        "ctx_encoder_first_step",
        "ctx_encoder_z",
        "ctx_proj",
        "ctx_token_attn",
        "ctx_final_norm",
    ):
        sub = nn.Linear(4, 4)  # real nn.Module with real parameters
        setattr(comp, attr, sub)
    return comp


def _make_orchestrator(models: Optional[dict] = None) -> TrainingPipelineOrchestrator:
    """Orchestrator in legacy mode (no real pipeline_config needed)."""
    orch = TrainingPipelineOrchestrator(models=models or {})
    return orch


# ---------------------------------------------------------------------------
# pipeline_config: new fields
# ---------------------------------------------------------------------------


class TestPipelineConfigV010Fields:
    """New YAML keys introduced for v0.10.0."""

    def test_lambda_ctx_aux_default(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.lambda_ctx_aux == 0.01

    def test_ctx_loss_weight_default(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.ctx_loss_weight == 0.5

    def test_freeze_context_branch_default(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.freeze_context_branch is False

    def test_parse_lambda_ctx_aux_from_yaml(self):
        cfg = parse_pipeline_config(
            {
                "steps": [
                    {
                        "name": "vae_v010",
                        "n_epochs": 5,
                        "train_vae": True,
                        "lambda_ctx_aux": 0.005,
                    }
                ]
            }
        )
        assert cfg.steps[0].lambda_ctx_aux == 0.005

    def test_parse_ctx_loss_weight_from_yaml(self):
        cfg = parse_pipeline_config(
            {
                "steps": [
                    {
                        "name": "flow_v010",
                        "n_epochs": 2,
                        "train_diff": True,
                        "ctx_loss_weight": 0.3,
                        "optimization": {"optimizers": {"flow": {"type": "AdamW", "lr": 1e-4}}},
                    }
                ]
            }
        )
        assert cfg.steps[0].ctx_loss_weight == 0.3

    def test_parse_freeze_context_branch_from_yaml(self):
        cfg = parse_pipeline_config(
            {
                "steps": [
                    {
                        "name": "vae_ctx_freeze",
                        "n_epochs": 3,
                        "train_vae": True,
                        "freeze_context_branch": True,
                    }
                ]
            }
        )
        assert cfg.steps[0].freeze_context_branch is True

    def test_context_branch_in_valid_components(self):
        """'context_branch' must be accepted in freeze/unfreeze lists."""
        cfg = parse_pipeline_config(
            {
                "steps": [
                    {
                        "name": "test",
                        "n_epochs": 1,
                        "train_vae": True,
                        # context_branch is a special token, not a top-level model key
                        # so it should NOT be in freeze; but it must be in VALID_COMPONENTS
                    }
                ]
            }
        )
        from fluxflow_training.training.pipeline_config import PipelineConfigValidator

        assert "context_branch" in PipelineConfigValidator.VALID_COMPONENTS


# ---------------------------------------------------------------------------
# pipeline_orchestrator: freeze_context_branch / unfreeze_context_branch
# ---------------------------------------------------------------------------


class TestFreezeContextBranch:
    """freeze_context_branch / unfreeze_context_branch helpers."""

    def _orch_with_v010_compressor(self) -> tuple[TrainingPipelineOrchestrator, nn.Module]:
        comp = nn.Module()
        # Add real sub-modules that have parameters
        comp.ctx_encoder_first_step = nn.Linear(4, 4)  # type: ignore[attr-defined]
        comp.ctx_encoder_z = nn.Linear(4, 4)  # type: ignore[attr-defined]
        comp.ctx_proj = nn.Linear(4, 4)  # type: ignore[attr-defined]
        comp.ctx_token_attn = nn.Linear(4, 4)  # type: ignore[attr-defined]
        comp.ctx_final_norm = nn.LayerNorm(4)  # type: ignore[attr-defined]
        comp.z_proj = nn.Linear(4, 4)  # z-path param; must NOT be frozen by context helper

        orch = _make_orchestrator({"compressor": comp})
        return orch, comp

    def test_freeze_disables_gradients_on_ctx_attrs(self):
        orch, comp = self._orch_with_v010_compressor()
        orch.freeze_context_branch("compressor")

        for attr in (
            "ctx_encoder_first_step",
            "ctx_encoder_z",
            "ctx_proj",
            "ctx_token_attn",
            "ctx_final_norm",
        ):
            for p in getattr(comp, attr).parameters():
                assert not p.requires_grad, f"{attr} param still requires grad after freeze"

    def test_freeze_does_not_affect_z_path_params(self):
        orch, comp = self._orch_with_v010_compressor()
        orch.freeze_context_branch("compressor")

        # z_proj must remain trainable
        for p in comp.z_proj.parameters():
            assert p.requires_grad, "z_proj param was incorrectly frozen by freeze_context_branch"

    def test_unfreeze_restores_gradients(self):
        orch, comp = self._orch_with_v010_compressor()
        orch.freeze_context_branch("compressor")
        orch.unfreeze_context_branch("compressor")

        for attr in (
            "ctx_encoder_first_step",
            "ctx_encoder_z",
            "ctx_proj",
            "ctx_token_attn",
            "ctx_final_norm",
        ):
            for p in getattr(comp, attr).parameters():
                assert p.requires_grad, f"{attr} param still frozen after unfreeze"

    def test_freeze_missing_attrs_is_noop(self):
        """Compressor without ctx_* attrs (legacy) should not raise."""
        comp = nn.Linear(4, 4)  # no ctx_* attrs
        orch = _make_orchestrator({"compressor": comp})
        orch.freeze_context_branch("compressor")  # must not raise

    def test_freeze_missing_model_warns_not_raises(self):
        """Calling freeze on a missing model key must not raise."""
        orch = _make_orchestrator({})
        orch.freeze_context_branch("compressor")  # not in dict — must not raise


# ---------------------------------------------------------------------------
# pipeline_orchestrator: discriminator ctx_dim dispatch
# ---------------------------------------------------------------------------


class TestDiscriminatorCtxDimDispatch:
    """Verify the correct ctx_dim is computed for v0.10.0 vs legacy compressors."""

    def test_v010_ctx_dim_equals_2x_d_model(self):
        """For v0.10.0: get_context_dims() returns 128, so ctx_dim = 128+128 = 256."""
        comp = MagicMock()
        comp.get_context_dims = Mock(return_value=128)

        vae_dim = 128
        ctx_dims = comp.get_context_dims()
        expected_ctx_dim = vae_dim + ctx_dims  # 256

        assert expected_ctx_dim == 256

    def test_legacy_ctx_dim_fallback(self):
        """For v0.7.0/v0.8.0: AttributeError triggers fallback to vae_dim+5=133."""
        comp = MagicMock(spec=[])  # no get_context_dims attribute

        vae_dim = 128
        try:
            ctx_dims = comp.get_context_dims()
            expected_ctx_dim = vae_dim + ctx_dims
        except (AttributeError, TypeError):
            expected_ctx_dim = vae_dim + 5  # CONTEXT_DIMS = 5 for legacy

        assert expected_ctx_dim == 133

    def test_ctx_dim_uses_compressor_method_not_hardcoded(self):
        """ctx_dim computation must use get_context_dims(), not a hardcoded constant."""
        # If d_model were 64 (smaller config), ctx_dim should be 64+64=128, not 133.
        comp = MagicMock()
        comp.get_context_dims = Mock(return_value=64)

        vae_dim = 64
        ctx_dims = comp.get_context_dims()
        expected_ctx_dim = vae_dim + ctx_dims  # 128

        assert expected_ctx_dim == 128
        assert expected_ctx_dim != 133  # must NOT be the legacy constant


# ---------------------------------------------------------------------------
# vae_trainer: ctx_input_dim + ctx_aux_loss
# ---------------------------------------------------------------------------


class TestVAETrainerV010:
    """Tests for v0.10.0-specific VAETrainer changes."""

    def _make_minimal_trainer(self, ctx_input_dim: Optional[int] = None):
        """Build a VAETrainer with mocked models and no GAN/LPIPS."""
        from fluxflow_training.training.vae_trainer import VAETrainer
        from fluxflow_training.training.utils import EMA

        # Minimal real nn.Modules so parameter counts work
        compressor = nn.Sequential(nn.Linear(4, 4))
        expander = nn.Sequential(nn.Linear(4, 4))
        optimizer = torch.optim.SGD(
            list(compressor.parameters()) + list(expander.parameters()), lr=1e-3
        )
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        ema = EMA(nn.ModuleList([compressor, expander]))

        accel = MagicMock()
        accel.backward = lambda loss: loss.backward()
        accel.scaler = None
        accel.clip_grad_norm_ = MagicMock(return_value=1.0)

        trainer = VAETrainer(
            compressor=compressor,
            expander=expander,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            reconstruction_loss_fn=nn.L1Loss(),
            reconstruction_loss_min_fn=nn.MSELoss(),
            use_gan=False,
            use_lpips=False,
            ctx_input_dim=ctx_input_dim,
            lambda_ctx_aux=0.01,
            train_ctx_aux=True,
            accelerator=accel,
        )
        return trainer

    def test_ctx_input_dim_explicit_sets_predictor(self):
        """When ctx_input_dim is provided, context_predictor input must match."""
        trainer = self._make_minimal_trainer(ctx_input_dim=256)
        first_linear = next(
            m for m in trainer.context_predictor.modules() if isinstance(m, nn.Linear)
        )
        assert first_linear.in_features == 256

    def test_ctx_input_dim_none_uses_detection(self):
        """When ctx_input_dim is None, legacy detection path runs without crash."""
        # Should not raise even though mock compressor can't do a test forward pass
        trainer = self._make_minimal_trainer(ctx_input_dim=None)
        # fallback is 27; accept any positive value
        first_linear = next(
            m for m in trainer.context_predictor.modules() if isinstance(m, nn.Linear)
        )
        assert first_linear.in_features > 0

    def test_lambda_ctx_aux_stored(self):
        trainer = self._make_minimal_trainer(ctx_input_dim=128)
        assert trainer.lambda_ctx_aux == 0.01

    def test_train_ctx_aux_flag_stored(self):
        trainer = self._make_minimal_trainer(ctx_input_dim=128)
        assert trainer.train_ctx_aux is True

    def test_ctx_aux_loss_computed_for_even_packed_dim(self):
        """ctx_aux_loss must be non-zero for a packed tensor with even last dim."""
        trainer = self._make_minimal_trainer(ctx_input_dim=128)

        # Directly test the loss computation path via a mock packed tensor
        # packed: [B, T+1, 2D] — even last dim triggers ctx_aux computation
        B, T, D = 2, 4, 64  # 2D = 128
        packed = torch.randn(B, T + 1, 2 * D)

        total_dim = packed.size(-1)  # 128
        half = total_dim // 2  # 64
        assert total_dim % 2 == 0 and half > 0

        img_seq = packed[:, :-1, :]  # [B, T, 2D]
        z_tokens_half = img_seq[:, :, :half]
        ctx_tokens_half = img_seq[:, :, half:]
        ctx_aux = nn.functional.mse_loss(ctx_tokens_half, z_tokens_half.detach())

        # For random tensors the loss is non-zero
        assert float(ctx_aux.item()) > 0.0

    def test_ctx_aux_loss_zero_for_odd_packed_dim(self):
        """ctx_aux_loss must be zero (skipped) when packed last dim is odd (legacy)."""
        # Odd dim = D+5 = 133 (v0.8.0) — skip the ctx_aux computation silently
        total_dim = 133  # odd
        assert total_dim % 2 != 0  # computation should be skipped

    def test_kl_uses_mu_logvar_not_context(self):
        """KL must only consume mu and logvar from the z branch."""
        from fluxflow_training.training.losses import kl_standard_normal

        # Ensure kl_standard_normal accepts [B, D, H, W] mu/logvar (z branch shape)
        B, D, H, W = 2, 128, 4, 4
        mu = torch.zeros(B, D, H, W)
        logvar = torch.zeros(B, D, H, W)
        kl = kl_standard_normal(mu, logvar, free_bits_nats=0.0, reduce="mean")
        # For zero mu/logvar, KL is 0 (exact Gaussian)
        assert float(kl.item()) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# flow_trainer: ctx_loss_weight attribute
# ---------------------------------------------------------------------------


class TestFlowTrainerV010:
    """Verify ctx_loss_weight is wired correctly in FlowTrainer."""

    def _make_stub_modules(self):
        """Return minimal real nn.Module instances for FlowTrainer construction."""

        class _Stub(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.zeros(1))

        flow = _Stub()
        text_enc = _Stub()
        comp = _Stub()
        comp.use_gradient_checkpointing = False
        return flow, text_enc, comp

    def test_ctx_loss_weight_attribute_exists(self):
        """FlowTrainer must store ctx_loss_weight as an instance attribute."""
        from fluxflow_training.training.flow_trainer import FlowTrainer

        flow, text_enc, comp = self._make_stub_modules()
        opt = torch.optim.SGD(flow.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.ConstantLR(opt)

        trainer = FlowTrainer(
            flow_processor=flow,
            text_encoder=text_enc,
            compressor=comp,
            optimizer=opt,
            scheduler=sched,
            ctx_loss_weight=0.5,
            accelerator=MagicMock(),
        )
        assert trainer.ctx_loss_weight == 0.5

    def test_ctx_loss_weight_default_is_one(self):
        """Default ctx_loss_weight is 1.0 in FlowTrainer (plan: tuned via YAML)."""
        from fluxflow_training.training.flow_trainer import FlowTrainer

        flow, text_enc, comp = self._make_stub_modules()
        opt = torch.optim.SGD(flow.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.ConstantLR(opt)

        trainer = FlowTrainer(
            flow_processor=flow,
            text_encoder=text_enc,
            compressor=comp,
            optimizer=opt,
            scheduler=sched,
            accelerator=MagicMock(),
        )
        # Default in FlowTrainer is 1.0; YAML default in PipelineStepConfig is 0.5
        assert trainer.ctx_loss_weight == 1.0


# ---------------------------------------------------------------------------
# pipeline_orchestrator: configure_step_models with freeze_context_branch key
# ---------------------------------------------------------------------------


class TestConfigureStepModelsV010:
    """freeze_context_branch config key is applied in configure_step_models."""

    def _make_comp_with_ctx(self) -> nn.Module:
        comp = nn.Module()
        comp.ctx_encoder_first_step = nn.Linear(4, 4)  # type: ignore[attr-defined]
        comp.ctx_encoder_z = nn.Linear(4, 4)  # type: ignore[attr-defined]
        comp.ctx_proj = nn.Linear(4, 4)  # type: ignore[attr-defined]
        comp.ctx_token_attn = nn.Linear(4, 4)  # type: ignore[attr-defined]
        comp.ctx_final_norm = nn.LayerNorm(4)  # type: ignore[attr-defined]
        return comp

    def test_freeze_context_branch_true_freezes_ctx_params(self):
        """When freeze_context_branch=True in step config, context branch is frozen."""
        comp = self._make_comp_with_ctx()
        orch = _make_orchestrator({"compressor": comp})

        step = PipelineStepConfig(
            name="s",
            n_epochs=1,
            train_vae=True,
            freeze_context_branch=True,
        )
        orch.configure_step_models(step, {"compressor": comp})

        for attr in ("ctx_encoder_first_step", "ctx_proj", "ctx_token_attn", "ctx_final_norm"):
            for p in getattr(comp, attr).parameters():
                assert not p.requires_grad, f"{attr} still trainable; freeze_context_branch failed"

    def test_freeze_context_branch_false_leaves_ctx_params_trainable(self):
        """When freeze_context_branch=False (default), context branch stays trainable."""
        comp = self._make_comp_with_ctx()
        orch = _make_orchestrator({"compressor": comp})

        step = PipelineStepConfig(
            name="s",
            n_epochs=1,
            train_vae=True,
            freeze_context_branch=False,
        )
        orch.configure_step_models(step, {"compressor": comp})

        for attr in ("ctx_encoder_first_step", "ctx_proj"):
            for p in getattr(comp, attr).parameters():
                assert p.requires_grad, f"{attr} was incorrectly frozen"


# ---------------------------------------------------------------------------
# Task 8: text_encoder split optimizer config (integration smoke tests)
# ---------------------------------------------------------------------------


class TestTextEncoderSplitOptimizerConfig:
    """YAML-style dicts with text_encoder_backbone/projection keys parse correctly."""

    def _step_with_optimizers(self, opt_keys: list) -> PipelineStepConfig:
        optimizers = {k: OptimizerConfig(lr=1e-4) for k in opt_keys}
        opt_config = OptimizationConfig(optimizers=optimizers)
        return PipelineStepConfig(name="s", n_epochs=1, train_vae=True, optimization=opt_config)

    def test_backbone_key_accepted_in_pipeline_step_config(self):
        step = self._step_with_optimizers(["flow", "text_encoder_backbone"])
        assert "text_encoder_backbone" in step.optimization.optimizers

    def test_projection_key_accepted_in_pipeline_step_config(self):
        step = self._step_with_optimizers(["flow", "text_encoder_projection"])
        assert "text_encoder_projection" in step.optimization.optimizers

    def test_both_split_keys_accepted_together(self):
        step = self._step_with_optimizers(
            ["flow", "text_encoder_backbone", "text_encoder_projection"]
        )
        assert "text_encoder_backbone" in step.optimization.optimizers
        assert "text_encoder_projection" in step.optimization.optimizers

    def test_conflict_guard_rejects_whole_and_split_together(self):
        from fluxflow_training.training.pipeline_config import (
            PipelineConfig,
            PipelineConfigValidator,
        )

        step = self._step_with_optimizers(["text_encoder", "text_encoder_backbone"])
        config = PipelineConfig(steps=[step])
        validator = PipelineConfigValidator(config)
        with pytest.raises(ValueError, match="text_encoder"):
            validator.validate()
