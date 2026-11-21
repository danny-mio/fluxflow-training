"""v0.10.0: automatic latent-stat calibration at the VAE -> flow step transition."""

import logging
from unittest.mock import MagicMock

import pytest
import safetensors.torch
import torch
import torch.nn as nn
from fluxflow.models.pipeline import FluxPipeline
from fluxflow.models.v100.vae import FluxCompressor_v100, FluxExpander_v100

from fluxflow_training.training.checkpoint_manager import CheckpointManager
from fluxflow_training.training.pipeline_config import (
    PipelineConfig,
    PipelineStepConfig,
    parse_pipeline_config,
)
from fluxflow_training.training.pipeline_orchestrator import TrainingPipelineOrchestrator


class _CountingCompressor(FluxCompressor_v100):
    """Tiny v100 compressor that counts forward calls."""

    def __init__(self) -> None:
        super().__init__(d_model=8, downscales=2, use_gradient_checkpointing=False)
        self.calls = 0

    def forward(self, img, training=False, sample=None):  # type: ignore[override]
        self.calls += 1
        return super().forward(img, training=training, sample=sample)


class _Loader:
    """Minimal dataloader: yields ([res tensors], input_ids); exposes ``dataset``."""

    def __init__(self, n_batches: int = 6, batch_size: int = 2) -> None:
        torch.manual_seed(0)
        self.batches = [
            (
                [
                    torch.rand(batch_size, 3, 32, 32) * 2 - 1,
                    torch.rand(batch_size, 3, 16, 16) * 2 - 1,
                ],
                torch.zeros(batch_size, 4, dtype=torch.long),
            )
            for _ in range(n_batches)
        ]
        self.dataset = list(range(n_batches * batch_size))

    def __iter__(self):
        return iter(self.batches)


class _StatefulSampler:
    def __init__(self) -> None:
        self.position = 0

    def state_dict(self) -> dict:
        return {"position": self.position}

    def load_state_dict(self, state: dict) -> None:
        self.position = state["position"]


def _models() -> dict:
    comp = _CountingCompressor()
    exp = FluxExpander_v100(d_model=8, upscales=2, use_gradient_checkpointing=False)
    flow = nn.Linear(2, 2)
    return {
        "diffuser": FluxPipeline(comp, flow, exp),
        "compressor": comp,
        "expander": exp,
        "flow_processor": flow,
        "text_encoder": nn.Linear(2, 2),
        "model_config": None,
    }


def _orch(tmp_path, steps) -> TrainingPipelineOrchestrator:
    return TrainingPipelineOrchestrator(
        pipeline_config=PipelineConfig(steps=steps),
        checkpoint_manager=CheckpointManager(output_dir=tmp_path),
        accelerator=None,
        device=torch.device("cpu"),
    )


def _vae_step(**kw) -> PipelineStepConfig:
    return PipelineStepConfig(name="vae", n_epochs=1, train_vae=True, **kw)


def _flow_step(name: str = "flow", **kw) -> PipelineStepConfig:
    return PipelineStepConfig(name=name, n_epochs=1, train_diff=True, **kw)


class TestStepConfig:
    def test_default_and_parse(self):
        assert PipelineStepConfig(name="s", n_epochs=1).calibration_images == 2000
        cfg = parse_pipeline_config(
            {"steps": [{"name": "f", "n_epochs": 1, "train_diff": True, "calibration_images": 64}]}
        )
        assert cfg.steps[0].calibration_images == 64

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="calibration_images"):
            parse_pipeline_config(
                {
                    "steps": [
                        {"name": "f", "n_epochs": 1, "train_diff": True, "calibration_images": -1}
                    ]
                }
            )


class TestMaybeCalibrate:
    def test_transition_calibrates_exactly_once_and_saves(self, tmp_path, caplog):
        steps = [_vae_step(), _flow_step("flow1"), _flow_step("flow2")]
        orch = _orch(tmp_path, steps)
        models = _models()
        comp = models["compressor"]
        loader = _Loader()

        assert orch._maybe_calibrate_latents(steps[0], models, loader, 2) is False
        assert comp.calls == 0 and not comp.latent_stats_calibrated

        with caplog.at_level(logging.INFO):
            assert orch._maybe_calibrate_latents(steps[1], models, loader, 2) is True
        assert comp.calls > 0
        assert models["diffuser"].latent_stats_calibrated
        assert not torch.equal(comp.latent_scale, torch.ones_like(comp.latent_scale))
        assert "latent_scale" in caplog.text or "std" in caplog.text  # per-channel table

        sd = safetensors.torch.load_file(str(orch.checkpoint_manager.model_path))
        for mod in ("compressor", "expander"):
            for buf in ("latent_shift", "latent_scale"):
                assert torch.equal(sd[f"diffuser.{mod}.{buf}"], getattr(models[mod], buf)), (
                    mod,
                    buf,
                )
        assert not torch.equal(
            sd["diffuser.compressor.latent_scale"], torch.ones_like(comp.latent_scale)
        )

        calls = comp.calls
        caplog.clear()
        with caplog.at_level(logging.INFO):
            assert orch._maybe_calibrate_latents(steps[2], models, loader, 2) is False
        assert comp.calls == calls
        assert "latent stats already calibrated" in caplog.text

    def test_train_diff_full_also_triggers(self, tmp_path):
        step = PipelineStepConfig(name="f", n_epochs=1, train_diff_full=True)
        orch = _orch(tmp_path, [step])
        models = _models()
        assert orch._maybe_calibrate_latents(step, models, _Loader(), 2) is True

    def test_max_batches_from_calibration_images(self, tmp_path):
        step = _flow_step(calibration_images=3)
        orch = _orch(tmp_path, [step])
        models = _models()
        orch._maybe_calibrate_latents(step, models, _Loader(), 2)
        assert models["compressor"].calls == 2 * 2  # ceil(3/2) batches x 2 resolutions

    def test_zero_disables_and_flow_trainer_then_raises(self, tmp_path):
        step = _flow_step(calibration_images=0)
        orch = _orch(tmp_path, [step])
        models = _models()
        assert orch._maybe_calibrate_latents(step, models, _Loader(), 2) is False
        assert models["compressor"].calls == 0
        assert not models["compressor"].latent_stats_calibrated
        assert not orch.checkpoint_manager.model_path.exists()

        from fluxflow_training.training import FlowTrainer

        opt = torch.optim.AdamW(models["flow_processor"].parameters(), lr=1e-4)
        with pytest.raises(ValueError, match="not calibrated"):
            FlowTrainer(
                flow_processor=models["flow_processor"],
                text_encoder=models["text_encoder"],
                compressor=models["compressor"],
                optimizer=opt,
                scheduler=None,
            )

    def test_legacy_compressor_skipped(self, tmp_path):
        step = _flow_step()
        orch = _orch(tmp_path, [step])
        models = _models()
        models["compressor"] = nn.Linear(2, 2)
        assert orch._maybe_calibrate_latents(step, models, _Loader(), 2) is False

    def test_restores_train_mode_and_sampler_state(self, tmp_path):
        step = _flow_step()
        orch = _orch(tmp_path, [step])
        models = _models()
        models["compressor"].train()
        sampler = _StatefulSampler()
        sampler.position = 7

        class _AdvancingLoader(_Loader):
            def __iter__(self):
                for b in self.batches:
                    sampler.position += 1
                    yield b

        orch._maybe_calibrate_latents(step, models, _AdvancingLoader(), 2, sampler=sampler)
        assert models["compressor"].training is True
        assert sampler.position == 7


class _Stop(Exception):
    pass


class TestRunOrdering:
    """Calibration must happen inside run() before EMA and FlowTrainer are created."""

    @pytest.mark.parametrize("joint", [False, True])
    def test_calibrated_before_ema_and_trainers(self, tmp_path, monkeypatch, joint):
        step = (
            PipelineStepConfig(name="j", n_epochs=1, train_vae=True, train_diff=True)
            if joint
            else _flow_step()
        )
        orch = _orch(tmp_path, [step])
        models = _models()
        seen = {}

        def _trainers(step_, models_, optimizers, schedulers, ema, args):
            seen["calibrated"] = models_["diffuser"].latent_stats_calibrated
            if ema is not None:
                seen["ema_shift"] = ema.shadow["0.latent_shift"].clone()
                seen["ema_scale"] = ema.shadow["0.latent_scale"].clone()
            raise _Stop

        monkeypatch.setattr(orch, "_generate_samples", lambda *a, **k: None)
        monkeypatch.setattr(orch, "configure_step_models", lambda *a, **k: None)
        monkeypatch.setattr(orch, "_ensure_discriminator_spectral_norm", lambda *a, **k: None)
        monkeypatch.setattr(orch, "_create_step_optimizers", lambda *a, **k: {})
        monkeypatch.setattr(orch, "_create_step_schedulers", lambda *a, **k: {})
        monkeypatch.setattr(orch, "_create_step_trainers", _trainers)

        args = MagicMock()
        args.batch_size = 2
        with pytest.raises(_Stop):
            orch.run(models, _Loader(), None, MagicMock(), MagicMock(), args, {})
        assert seen["calibrated"] is True
        if joint:
            comp = models["compressor"]
            assert torch.equal(seen["ema_shift"], comp.latent_shift)
            assert torch.equal(seen["ema_scale"], comp.latent_scale)
