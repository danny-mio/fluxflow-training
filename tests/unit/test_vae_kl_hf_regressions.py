"""KL default/warning/warmup units and HF-ramp length (fixed hf_ramp_steps)."""

import inspect
import logging

import pytest
import torch.nn as nn

from fluxflow_training.training.pipeline_config import PipelineStepConfig, parse_pipeline_config
from fluxflow_training.training.pipeline_orchestrator import TrainingPipelineOrchestrator
from fluxflow_training.training.vae_trainer import VAETrainer
from tests.unit.test_vae_trainer_audit_fixes import _imgs, _mk

LOGGER = "fluxflow_training.training.vae_trainer"


def _wrap(step):
    return {"steps": [step]}


def _step(**kw):
    cfg = parse_pipeline_config(_wrap({"name": "s", "n_epochs": 1, "train_vae": True, **kw}))
    return cfg.steps[0]


class TestKlDefault:
    def test_dataclass_default(self):
        assert PipelineStepConfig(name="s", n_epochs=1).kl_z_weight == 1e-3

    def test_parse_default(self):
        assert _step().kl_z_weight == 1e-3


class TestKlWarning:
    def _v100(self, **kw):
        t = _mk(train_kl=True, **kw)
        t.compressor.ctx_zinject_norm = nn.Identity()
        return t

    def test_fires_once_for_large_weight(self, caplog):
        t = self._v100(kl_z_weight=0.5)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            for i in range(3):
                t.train_step(_imgs(), global_step=i)
        msgs = [r for r in caplog.records if "kl_z_weight" in r.getMessage()]
        assert len(msgs) == 1 and "per-element" in msgs[0].getMessage()

    def test_silent_for_small_weight(self, caplog):
        t = self._v100(kl_z_weight=1e-3)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            t.train_step(_imgs(), global_step=0)
        assert not [r for r in caplog.records if "kl_z_weight" in r.getMessage()]


class TestKlWarmupUnits:
    def test_full_weight_after_warmup_times_accum(self):
        t = _mk(
            train_kl=True, kl_z_weight=0.01, kl_z_warmup_steps=10, gradient_accumulation_steps=16
        )
        assert t._kl_z_beta(159) < 0.01
        assert t._kl_z_beta(160) == pytest.approx(0.01)
        assert 0 < t._kl_z_beta(80) < 0.01

    def test_accum_one_unchanged(self):
        t = _mk(train_kl=True, kl_z_weight=0.01, kl_z_warmup_steps=10)
        assert t._kl_z_beta(10) == pytest.approx(0.01)

    def test_reported_kl_beta_uses_same_schedule(self):
        t = _mk(train_kl=True, kl_z_weight=0.01, kl_z_warmup_steps=2, gradient_accumulation_steps=2)
        out = t.train_step(_imgs(), global_step=3)
        assert out["kl_beta"] < 0.01
        out = t.train_step(_imgs(), global_step=4)
        assert out["kl_beta"] == pytest.approx(0.01)


class TestHfRamp:
    def _recon(self, ramp, step):
        import torch

        torch.manual_seed(0)
        return _mk(hf_ramp_steps=ramp)._train_generator(_imgs(), step)["recon"]

    def test_default(self):
        assert inspect.signature(VAETrainer.__init__).parameters["hf_ramp_steps"].default == 30000
        assert not hasattr(_mk(), "total_training_steps")

    def test_ramp_is_fixed_fraction_of_hf_ramp_steps(self):
        assert self._recon(10, 5) == pytest.approx(self._recon(100, 50))
        assert self._recon(10, 5) != pytest.approx(self._recon(10, 10))

    def test_saturates_at_ramp_end(self):
        assert self._recon(10, 10) == pytest.approx(self._recon(10, 999))

    def test_zero_ramp_is_full_alpha(self):
        assert self._recon(0, 0) == pytest.approx(self._recon(10, 10))


class TestHfRampWiring:
    def test_config_default_and_parse(self):
        assert PipelineStepConfig(name="s", n_epochs=1).hf_ramp_steps == 30000
        assert _step(hf_ramp_steps=500).hf_ramp_steps == 500

    def test_config_rejects_negative(self):
        with pytest.raises(ValueError, match="hf_ramp_steps"):
            _step(hf_ramp_steps=-1)

    def test_orchestrator_forwards(self):
        from tests.unit.test_pipeline_orchestrator_audit_fixes import (
            _V100Comp,
            _args,
            _gan_step,
            _orch,
        )
        from unittest.mock import MagicMock, patch

        step = _gan_step(hf_ramp_steps=1234)
        models = {"compressor": _V100Comp(), "expander": nn.Linear(1, 1), "D_img": None}
        opts = {"vae": MagicMock(), "discriminator": MagicMock()}
        with patch("fluxflow_training.training.VAETrainer") as vt:
            _orch([step])._create_step_trainers(step, models, opts, {}, None, _args())
        kw = vt.call_args.kwargs
        assert kw["hf_ramp_steps"] == 1234
        assert "total_training_steps" not in kw
        assert (
            "total_steps"
            not in inspect.signature(TrainingPipelineOrchestrator._create_step_trainers).parameters
        )
