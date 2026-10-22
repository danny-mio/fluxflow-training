"""Audit fixes in TrainingPipelineOrchestrator: kwarg wiring (A2), scheduler budget (A3),
D spectral-norm / ctx_dim (C5/C6), D optimizer defaults + activation WD exemption (C7/E7),
EMA applied for exports and samples (D3)."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from fluxflow.models import PatchDiscriminator, TrainableBezier

from fluxflow_training.training.pipeline_config import (
    OptimizationConfig,
    OptimizerConfig,
    PipelineConfig,
    PipelineStepConfig,
)
from fluxflow_training.training.pipeline_orchestrator import (
    TrainingPipelineOrchestrator,
    compute_total_global_steps,
)
from fluxflow_training.training.utils import EMA, has_spectral_norm, record_gan_losses


def _orch(steps=None) -> TrainingPipelineOrchestrator:
    orch = TrainingPipelineOrchestrator.__new__(TrainingPipelineOrchestrator)
    orch.config = PipelineConfig(steps=steps or [])
    orch.device = "cpu"
    orch.accelerator = MagicMock()
    orch.global_step = 5
    orch.steps_completed = []
    return orch


def _args(**kw):
    base = dict(
        initial_clipping_norm=1.0,
        feature_maps_dim_disc=4,
        channels=3,
        discriminator_use_spectral_norm=True,
        lr=1e-4,
        no_samples=False,
        test_image_address=["x.png"],
        output_path="out",
        sample_captions=[],
        batch_size=1,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class _V100Comp(nn.Module):
    d_model = 8

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.act = TrainableBezier((2,), channel_only=True)
        self.ctx_zinject_norm = nn.GroupNorm(1, 2, affine=False)

    def get_context_dims(self):
        return 8


def _gan_step(**kw) -> PipelineStepConfig:
    return PipelineStepConfig(name="g", n_epochs=1, train_vae=True, gan_training=True, **kw)


class TestComputeTotalGlobalSteps:
    def test_plain(self):
        assert compute_total_global_steps(3, 10, None) == 30

    def test_max_steps_caps_epoch_batches(self):
        assert compute_total_global_steps(3, 10, 4) == 12
        assert compute_total_global_steps(3, 10, 50) == 30

    def test_never_zero(self):
        assert compute_total_global_steps(1, 0, None) == 1


class TestVaeTrainerKwargWiring:
    def _kwargs(self, step, total_steps=123):
        comp = _V100Comp()
        models = {"compressor": comp, "expander": nn.Linear(1, 1), "D_img": None}
        opts = {"vae": MagicMock(), "discriminator": MagicMock()}
        with patch("fluxflow_training.training.VAETrainer") as vt:
            _orch([step])._create_step_trainers(
                step, models, opts, {}, None, _args(), total_steps=total_steps
            )
        return vt.call_args.kwargs

    def test_previously_dropped_keys_reach_trainer(self):
        kw = self._kwargs(
            _gan_step(
                kl_z_warmup_steps=77,
                ctx_shrinkage_warmup_start_step=11,
                ctx_shrinkage_warmup_steps=22,
                ctx_shrinkage_max_mean_sq=33.0,
                gan_start_step=3,
                gan_warmup_steps=4,
            )
        )
        assert kw["kl_z_warmup_steps"] == 77
        assert kw["ctx_shrinkage_warmup_start_step"] == 11
        assert kw["ctx_shrinkage_warmup_steps"] == 22
        assert kw["ctx_shrinkage_max_mean_sq"] == 33.0
        assert (kw["gan_start_step"], kw["gan_warmup_steps"]) == (3, 4)
        assert kw["total_training_steps"] == 123

    def test_new_defaults(self):
        kw = self._kwargs(_gan_step())
        assert kw["train_ctx_aux"] is False
        assert kw["instance_noise_std"] == 0.05
        assert (kw["gan_start_step"], kw["gan_warmup_steps"]) == (0, 1000)

    def test_discriminator_built_without_ctx_proj_for_v100(self):
        step = _gan_step()
        models = {"compressor": _V100Comp(), "expander": nn.Linear(1, 1), "D_img": None}
        with patch("fluxflow_training.training.VAETrainer"):
            _orch([step])._create_step_trainers(
                step, models, {"vae": MagicMock()}, {}, None, _args()
            )
        assert models["D_img"].ctx_dim == 0
        assert not hasattr(models["D_img"], "ctx_proj")

    def test_loaded_v100_discriminator_with_zero_ctx_dim_accepted(self):
        step = _gan_step()
        d = PatchDiscriminator(3, 4, 3, ctx_dim=0, use_spectral_norm=True)
        models = {"compressor": _V100Comp(), "expander": nn.Linear(1, 1), "D_img": d}
        with patch("fluxflow_training.training.VAETrainer"):
            _orch([step])._create_step_trainers(
                step, models, {"vae": MagicMock()}, {}, None, _args()
            )
        assert models["D_img"] is d


class TestEnsureDiscriminatorSpectralNorm:
    def _models(self, sn):
        return {"D_img": PatchDiscriminator(3, 4, 3, ctx_dim=0, use_spectral_norm=sn)}

    def test_step_flag_false_rebuilds_loudly(self, caplog):
        models = self._models(True)
        with caplog.at_level(logging.WARNING):
            _orch()._ensure_discriminator_spectral_norm(
                _gan_step(discriminator_use_spectral_norm=False), models, _args()
            )
        assert not has_spectral_norm(models["D_img"])
        assert models["D_img"].backbone[0].out_channels == 4
        assert any("spectral" in r.message.lower() for r in caplog.records)

    def test_unset_step_flag_inherits_cli(self):
        models = self._models(True)
        _orch()._ensure_discriminator_spectral_norm(
            _gan_step(), models, _args(discriminator_use_spectral_norm=False)
        )
        assert not has_spectral_norm(models["D_img"])

    def test_matching_flag_keeps_same_object(self):
        models = self._models(True)
        d = models["D_img"]
        _orch()._ensure_discriminator_spectral_norm(
            _gan_step(discriminator_use_spectral_norm=True), models, _args()
        )
        assert models["D_img"] is d

    def test_non_gan_step_untouched(self):
        models = self._models(True)
        d = models["D_img"]
        step = PipelineStepConfig(
            name="v", n_epochs=1, train_vae=True, discriminator_use_spectral_norm=False
        )
        _orch()._ensure_discriminator_spectral_norm(step, models, _args())
        assert models["D_img"] is d


class TestOptimizerCreation:
    def _models(self):
        return {
            "compressor": _V100Comp(),
            "expander": nn.Linear(1, 1),
            "D_img": PatchDiscriminator(3, 4, 3),
        }

    def test_default_branch_discriminator_uses_gan_defaults(self):
        opts = _orch()._create_step_optimizers(_gan_step(), self._models(), _args())
        assert opts["discriminator"].param_groups[0]["betas"] == (0.0, 0.9)

    def test_fallback_branch_discriminator_not_vae_config(self):
        step = _gan_step(
            optimization=OptimizationConfig(
                optimizers={"vae": OptimizerConfig(type="AdamW", lr=1e-4, betas=(0.9, 0.999))}
            )
        )
        opts = _orch()._create_step_optimizers(step, self._models(), _args())
        assert opts["discriminator"].param_groups[0]["betas"] == (0.0, 0.9)

    def test_fallback_branch_lion_vae_gets_lion_discriminator(self):
        step = _gan_step(
            optimization=OptimizationConfig(
                optimizers={"vae": OptimizerConfig(type="Lion", lr=1e-5, betas=(0.9, 0.99))}
            )
        )
        opts = _orch()._create_step_optimizers(step, self._models(), _args())
        assert type(opts["discriminator"]).__name__ == "Lion"

    @pytest.mark.parametrize("explicit", [False, True])
    def test_vae_activation_params_not_decayed(self, explicit):
        opt_cfg = (
            OptimizationConfig(optimizers={"vae": OptimizerConfig(type="AdamW", lr=1e-4)})
            if explicit
            else None
        )
        models = self._models()
        opts = _orch()._create_step_optimizers(_gan_step(optimization=opt_cfg), models, _args())
        nd = [g for g in opts["vae"].param_groups if g["weight_decay"] == 0.0]
        act = {id(p) for p in models["compressor"].act.parameters()}
        assert act and act <= {id(p) for g in nd for p in g["params"]}


class TestEmaAppliedOnExportAndSamples:
    def _setup(self):
        comp, exp = nn.Linear(2, 2), nn.Linear(2, 2)
        ema = EMA(nn.ModuleList([comp, exp]), decay=0.5)
        shadow = comp.weight.detach().clone()
        with torch.no_grad():
            comp.weight.add_(10.0)
        raw = comp.weight.detach().clone()
        models = {
            "diffuser": MagicMock(),
            "text_encoder": MagicMock(),
            "D_img": None,
            "compressor": comp,
        }
        return comp, ema, models, shadow, raw

    def test_save_models_sees_ema_weights_and_state_sees_raw(self):
        comp, ema, models, shadow, raw = self._setup()
        seen = {}
        orch = _orch([_gan_step()])
        orch.checkpoint_manager = MagicMock()
        orch.checkpoint_manager.save_models.side_effect = lambda **k: seen.setdefault(
            "models", comp.weight.detach().clone()
        )
        orch.checkpoint_manager.save_training_state.side_effect = lambda **k: seen.setdefault(
            "state", comp.weight.detach().clone()
        )
        orch.steps_completed = []
        orch._save_checkpoint(0, 0, 1, models, {}, {}, ema, _args())
        assert torch.equal(seen["models"], shadow)
        assert torch.equal(seen["state"], raw)
        assert torch.equal(comp.weight, raw)

    def test_raw_restored_if_save_fails(self):
        comp, ema, models, shadow, raw = self._setup()
        orch = _orch([_gan_step()])
        orch.checkpoint_manager = MagicMock()
        orch.checkpoint_manager.save_models.side_effect = OSError("disk")
        with pytest.raises(OSError):
            orch._save_checkpoint(0, 0, 1, models, {}, {}, ema, _args())
        assert torch.equal(comp.weight, raw)

    def test_samples_use_ema_weights_then_restore(self):
        comp, ema, models, shadow, raw = self._setup()
        seen = {}

        def fake_sample(*a, **k):
            seen["w"] = comp.weight.detach().clone()

        orch = _orch([_gan_step()])
        with patch("fluxflow_training.training.pipeline_orchestrator.safe_vae_sample", fake_sample):
            orch._generate_samples(_gan_step(), 0, 0, 1, models, None, _args(), [], ema=ema)
        assert torch.equal(seen["w"], shadow)
        assert torch.equal(comp.weight, raw)

    def test_samples_without_ema_still_work(self):
        _, _, models, _, _ = self._setup()
        with patch("fluxflow_training.training.pipeline_orchestrator.safe_vae_sample") as s:
            _orch([_gan_step()])._generate_samples(_gan_step(), 0, 0, 1, models, None, _args(), [])
        assert s.called


class TestRecordGanLosses:
    def test_fills_generator_buffer(self):
        from fluxflow_training.training.utils import FloatBuffer

        tr = SimpleNamespace(g_loss_buffer=FloatBuffer(5))
        record_gan_losses(tr, {"generator": 2.0})
        record_gan_losses(tr, {"vae": 1.0})
        record_gan_losses(tr, {"generator": 4.0})
        assert tr.g_loss_buffer.average == 3.0

    def test_trainer_without_buffer_is_ignored(self):
        record_gan_losses(SimpleNamespace(), {"generator": 1.0})
