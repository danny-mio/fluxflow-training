"""v0.10.0: automatic latent-stat calibration at the VAE -> flow step transition."""

import logging
import re
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
        # Per-channel table: one "<ch>  <part>  <mean>  <std>" row per channel + summaries
        rows = re.findall(r"^\s*(\d+)\s+(z|ctx)\s+-?\d+\.\d{4}\s+\d+\.\d{4}$", caplog.text, re.M)
        n_ch = comp.latent_shift.numel()
        assert [int(r[0]) for r in rows] == list(range(n_ch))
        n_ctx = comp.get_context_dims()
        assert [r[1] for r in rows] == ["z"] * (n_ch - n_ctx) + ["ctx"] * n_ctx
        assert re.search(r"z summary: \|mean\| max \d+\.\d{4}, std min", caplog.text)
        assert re.search(r"ctx summary: \|mean\| max \d+\.\d{4}, std min", caplog.text)

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


def _patch_run_scaffolding(monkeypatch, orch) -> None:
    monkeypatch.setattr(orch, "_generate_samples", lambda *a, **k: None)
    monkeypatch.setattr(orch, "configure_step_models", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_ensure_discriminator_spectral_norm", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_create_step_optimizers", lambda *a, **k: {})
    monkeypatch.setattr(orch, "_create_step_schedulers", lambda *a, **k: {})


def _run_args() -> MagicMock:
    args = MagicMock()
    args.batch_size = 2
    args.log_interval = 10
    args.checkpoint_save_interval = 10**9
    args.samples_per_checkpoint = 10**9
    args.no_samples = True
    return args


def _tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = 0
    return tok


def _write_resume_state(orch, step_idx, batch_idx, ema) -> None:
    """training_state.json + training_states.pt as _save_checkpoint would write them."""
    orch.checkpoint_manager.save_training_state(
        epoch=0,
        batch_idx=batch_idx,
        global_step=batch_idx,
        optimizers={},
        schedulers={},
        ema=ema,
        pipeline_metadata=orch.get_pipeline_metadata(step_idx, 0, batch_idx),
    )


class TestVaeEmaAdoptedByFlowStep:
    """Ruling: the flow step after a VAE step trains/calibrates on the EMA VAE."""

    def test_flow_step_uses_ema_weights_and_calibrates_on_them(self, tmp_path, monkeypatch):
        import fluxflow_training.training.pipeline_orchestrator as po
        from fluxflow_training.training.utils import EMA

        steps = [_vae_step(), _flow_step()]
        orch = _orch(tmp_path, steps)
        models = _models()
        comp, exp = models["compressor"], models["expander"]

        # VAE step state: EMA shadow differs from the raw weights.
        ema = EMA(nn.ModuleList([comp, exp]), decay=0.999)
        with torch.no_grad():
            for p in list(comp.parameters()) + list(exp.parameters()):
                p.add_(0.5)
        raw = [p.detach().clone() for p in comp.parameters()]
        shadow = {k: v.clone() for k, v in ema.shadow.items()}
        # Resume "beyond the end" of the VAE step so its loop is skipped (no VAE training).
        _write_resume_state(orch, 0, 10**6, ema)

        seen: dict = {}
        real_calibrate = po.calibrate_models

        def _spy_calibrate(models_, *a, **k):
            seen["calib_params"] = [p.detach().clone() for p in models_["compressor"].parameters()]
            return real_calibrate(models_, *a, **k)

        def _trainers(step_, models_, optimizers, schedulers, ema_, args):
            if step_.train_diff:
                seen["flow_params"] = [
                    p.detach().clone() for p in models_["compressor"].parameters()
                ]
                seen["calibrated"] = models_["diffuser"].latent_stats_calibrated
                raise _Stop
            return {}

        _patch_run_scaffolding(monkeypatch, orch)
        monkeypatch.setattr(po, "calibrate_models", _spy_calibrate)
        monkeypatch.setattr(orch, "_create_step_trainers", _trainers)
        with pytest.raises(_Stop):
            orch.run(models, _Loader(), None, _tokenizer(), MagicMock(), _run_args(), {})

        ema_params = [shadow[f"0.{n}"] for n, _ in comp.named_parameters()]
        for got, want, r in zip(seen["flow_params"], ema_params, raw):
            assert torch.equal(got, want)
            assert not torch.equal(got, r)
        for got, want in zip(seen["calib_params"], ema_params):
            assert torch.equal(got, want)  # stats computed AFTER the EMA copy
        assert seen["calibrated"] is True

    def test_no_copy_when_next_step_trains_vae(self, tmp_path):
        from fluxflow_training.training.utils import EMA

        orch = _orch(tmp_path, [_vae_step(), _vae_step()])
        models = _models()
        ema = EMA(nn.ModuleList([models["compressor"], models["expander"]]))
        with torch.no_grad():
            next(models["compressor"].parameters()).add_(1.0)
        raw = next(models["compressor"].parameters()).detach().clone()
        assert orch._adopt_vae_ema_for_next_step(ema, orch.config.steps[1]) is False
        assert orch._adopt_vae_ema_for_next_step(ema, None) is False
        assert orch._adopt_vae_ema_for_next_step(None, _flow_step()) is False
        assert torch.equal(next(models["compressor"].parameters()), raw)

    def test_copy_keeps_live_buffers(self, tmp_path):
        from tests.conftest import mark_latents_calibrated

        from fluxflow_training.training.utils import EMA

        orch = _orch(tmp_path, [_vae_step(), _flow_step()])
        models = _models()
        ema = EMA(nn.ModuleList([models["compressor"], models["expander"]]))
        mark_latents_calibrated(models["compressor"], models["expander"])
        assert orch._adopt_vae_ema_for_next_step(ema, orch.config.steps[1]) is True
        assert models["diffuser"].latent_stats_calibrated


class TestResumeCannotRevertStats:
    """Review Minor #1: ema_raw loaded on resume must not leave identity stats for flow."""

    def test_resume_with_identity_ema_raw_recalibrates_before_training(self, tmp_path, monkeypatch):
        from tests.conftest import mark_latents_calibrated

        from fluxflow_training.training.utils import EMA

        step = PipelineStepConfig(name="j", n_epochs=1, train_vae=True, train_diff=True)
        orch = _orch(tmp_path, [step])
        models = _models()
        comp, exp = models["compressor"], models["expander"]
        # Training state written before calibration: ema_raw holds identity stats.
        _write_resume_state(orch, 0, 2, EMA(nn.ModuleList([comp, exp])))
        # Models loaded from a calibrated export.
        mark_latents_calibrated(comp, exp)

        seen: dict = {}

        class _FakeFlow:
            ema = None

            def train_step(self, *a, **k):
                seen["calibrated"] = models["diffuser"].latent_stats_calibrated
                seen["scale"] = comp.latent_scale.clone()
                raise _Stop

        _patch_run_scaffolding(monkeypatch, orch)
        monkeypatch.setattr(orch, "_create_step_trainers", lambda *a, **k: {"flow": _FakeFlow()})
        with pytest.raises(_Stop):
            orch.run(models, _Loader(), None, _tokenizer(), MagicMock(), _run_args(), {})
        assert seen["calibrated"] is True
        assert not torch.equal(seen["scale"], torch.ones_like(seen["scale"]))

    def test_resume_reverted_with_calibration_disabled_raises(self, tmp_path, monkeypatch):
        from tests.conftest import mark_latents_calibrated

        from fluxflow_training.training.utils import EMA

        step = PipelineStepConfig(
            name="j", n_epochs=1, train_vae=True, train_diff=True, calibration_images=0
        )
        orch = _orch(tmp_path, [step])
        models = _models()
        comp, exp = models["compressor"], models["expander"]
        _write_resume_state(orch, 0, 2, EMA(nn.ModuleList([comp, exp])))
        mark_latents_calibrated(comp, exp)

        class _FakeFlow:
            ema = None

            def train_step(self, *a, **k):
                raise AssertionError("train_step must not run on uncalibrated latents")

        _patch_run_scaffolding(monkeypatch, orch)
        monkeypatch.setattr(orch, "_create_step_trainers", lambda *a, **k: {"flow": _FakeFlow()})
        with pytest.raises(ValueError, match="not calibrated"):
            orch.run(models, _Loader(), None, _tokenizer(), MagicMock(), _run_args(), {})
