"""Tests for CheckpointManager.save_models metadata whitelist.

Covers the activation_type (Bezier vs Padé) metadata fields specifically --
these must round-trip through the saved checkpoint's safetensors header so
other tools (fluxflow-ui, fluxflow-comfyui) can read back which activation
family a checkpoint was trained with.
"""

import safetensors.torch
import torch.nn as nn

from fluxflow_training.training.checkpoint_manager import CheckpointManager
from fluxflow_training.training.utils import EMA


def _fake_model_config(**overrides):
    base = {
        "model_type": "bezier",
        "model_version": "0.10.0",
        "vae_dim": 32,
        "activation_type": "bezier",
        "compressor_activation_type": None,
        "expander_activation_type": None,
        "flow_activation_type": None,
    }
    base.update(overrides)
    return base


def test_save_models_writes_activation_type_to_metadata(tmp_path):
    manager = CheckpointManager(output_dir=tmp_path)
    manager.save_models(
        diffuser=nn.Linear(4, 4),
        text_encoder=nn.Linear(4, 4),
        model_config=_fake_model_config(activation_type="pade"),
    )

    with safetensors.safe_open(str(manager.model_path), framework="pt", device="cpu") as f:
        metadata = f.metadata()

    assert metadata["activation_type"] == "pade"


def test_save_models_default_activation_type_is_bezier(tmp_path):
    manager = CheckpointManager(output_dir=tmp_path)
    manager.save_models(
        diffuser=nn.Linear(4, 4),
        text_encoder=nn.Linear(4, 4),
        model_config=_fake_model_config(activation_type="bezier"),
    )

    with safetensors.safe_open(str(manager.model_path), framework="pt", device="cpu") as f:
        metadata = f.metadata()

    assert metadata["activation_type"] == "bezier"


def test_save_models_writes_per_component_overrides_when_set(tmp_path):
    manager = CheckpointManager(output_dir=tmp_path)
    manager.save_models(
        diffuser=nn.Linear(4, 4),
        text_encoder=nn.Linear(4, 4),
        model_config=_fake_model_config(
            activation_type="bezier", compressor_activation_type="pade"
        ),
    )

    with safetensors.safe_open(str(manager.model_path), framework="pt", device="cpu") as f:
        metadata = f.metadata()

    assert metadata["compressor_activation_type"] == "pade"
    assert "expander_activation_type" not in metadata
    assert "flow_activation_type" not in metadata


def test_save_models_without_model_config_writes_no_activation_metadata(tmp_path):
    """Legacy call sites that pass model_config=None must not error and must
    write no activation_type key (mirrors existing model_type/vae_dim behavior)."""
    manager = CheckpointManager(output_dir=tmp_path)
    manager.save_models(
        diffuser=nn.Linear(4, 4),
        text_encoder=nn.Linear(4, 4),
        model_config=None,
    )

    with safetensors.safe_open(str(manager.model_path), framework="pt", device="cpu") as f:
        metadata = f.metadata()

    assert metadata is None or "activation_type" not in metadata


class TestEmaRawWeightsResume:
    """D3: exports carry EMA weights; training state keeps raw weights + EMA for exact resume."""

    def test_raw_weights_restored_on_resume(self, tmp_path):
        import torch
        import torch.nn as nn

        from fluxflow_training.training.checkpoint_manager import CheckpointManager
        from fluxflow_training.training.utils import EMA

        m = nn.Linear(2, 2)
        ema = EMA(m, decay=0.9)
        with torch.no_grad():
            m.weight.add_(1.0)
        ema.update()
        raw = m.weight.detach().clone()
        shadow = ema.shadow["weight"].clone()
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.save_training_state(
            epoch=0, batch_idx=1, global_step=1, optimizers={}, schedulers={}, ema=ema
        )

        m2 = nn.Linear(2, 2)  # fresh process; weights loaded from (EMA) export would differ
        ema2 = EMA(m2, decay=0.9)
        assert mgr.load_optimizer_scheduler_ema_states({}, {}, ema2)
        assert torch.equal(m2.weight, raw)
        assert torch.equal(ema2.shadow["weight"], shadow)

    def test_ema_state_ignored_when_ema_disabled(self, tmp_path):
        import torch.nn as nn

        from fluxflow_training.training.checkpoint_manager import CheckpointManager
        from fluxflow_training.training.utils import EMA

        ema = EMA(nn.Linear(2, 2))
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.save_training_state(
            epoch=0, batch_idx=1, global_step=1, optimizers={}, schedulers={}, ema=ema
        )
        assert mgr.load_optimizer_scheduler_ema_states({}, {}, None)


class TestFlowEmaPersistence:
    """Flow EMA lives under its own key and never collides with the VAE EMA."""

    def _emas(self):
        import torch

        vae, flow = nn.Linear(2, 2), nn.Linear(3, 3)
        e_vae, e_flow = EMA(vae, decay=0.9), EMA(flow, decay=0.9)
        with torch.no_grad():
            flow.weight.add_(1.0)
        e_flow.update()
        return e_vae, e_flow

    def test_roundtrip_both_emas(self, tmp_path):
        import torch

        e_vae, e_flow = self._emas()
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.save_training_state(
            epoch=0,
            batch_idx=1,
            global_step=1,
            optimizers={},
            schedulers={},
            ema=e_vae,
            ema_flow=e_flow,
        )
        states = torch.load(mgr.training_states_path, weights_only=True)
        assert "ema" in states and "ema_flow" in states and "ema_flow_raw" not in states
        n_vae, n_flow = EMA(nn.Linear(2, 2), decay=0.9), EMA(nn.Linear(3, 3), decay=0.9)
        assert mgr.load_optimizer_scheduler_ema_states({}, {}, n_vae, ema_flow=n_flow)
        assert torch.equal(n_flow.shadow["weight"], e_flow.shadow["weight"])
        assert torch.equal(n_vae.shadow["weight"], e_vae.shadow["weight"])

    def test_flow_only_without_vae_ema(self, tmp_path):
        import torch

        _, e_flow = self._emas()
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.save_training_state(
            epoch=0,
            batch_idx=1,
            global_step=1,
            optimizers={},
            schedulers={},
            ema=None,
            ema_flow=e_flow,
        )
        n_flow = EMA(nn.Linear(3, 3), decay=0.9)
        assert mgr.load_optimizer_scheduler_ema_states({}, {}, None, ema_flow=n_flow)
        assert torch.equal(n_flow.shadow["weight"], e_flow.shadow["weight"])

    def test_old_file_without_ema_flow_loads(self, tmp_path):
        import torch

        e_vae, _ = self._emas()
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.save_training_state(
            epoch=0, batch_idx=1, global_step=1, optimizers={}, schedulers={}, ema=e_vae
        )
        n_flow = EMA(nn.Linear(3, 3), decay=0.9)
        before = n_flow.shadow["weight"].clone()
        assert mgr.load_optimizer_scheduler_ema_states({}, {}, None, ema_flow=n_flow)
        assert torch.equal(n_flow.shadow["weight"], before)

    def test_ema_flow_ignored_when_not_requested(self, tmp_path):
        _, e_flow = self._emas()
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.save_training_state(
            epoch=0,
            batch_idx=1,
            global_step=1,
            optimizers={},
            schedulers={},
            ema=None,
            ema_flow=e_flow,
        )
        assert mgr.load_optimizer_scheduler_ema_states({}, {}, None)


class TestLatentStatsRoundTrip:
    """v0.10.0 latent_shift/latent_scale survive every training-side save/load path."""

    @staticmethod
    def _models():
        import torch
        from fluxflow.models.pipeline import FluxPipeline
        from fluxflow.models.v100.vae import FluxCompressor_v100, FluxExpander_v100

        comp = FluxCompressor_v100(d_model=8, downscales=2, use_gradient_checkpointing=False)
        exp = FluxExpander_v100(d_model=8, upscales=2, use_gradient_checkpointing=False)
        flow = nn.Linear(2, 2)
        diffuser = FluxPipeline(comp, flow, exp)
        return diffuser, comp, exp, flow, torch

    def _calibrated(self):
        diffuser, comp, exp, flow, torch = self._models()
        shift = torch.linspace(-1.0, 1.0, 16)
        scale = torch.linspace(0.5, 3.0, 16)
        diffuser.set_latent_stats(shift, scale)
        return diffuser, comp, exp, shift, scale

    def _load_like_initialize_models(self, mgr):
        """Mirror scripts/train.py initialize_models: load_models_parallel + strict=False."""
        diffuser, comp, exp, flow, _ = self._models()
        states, _ = mgr.load_models_parallel()
        comp.load_state_dict(states["diffuser.compressor"], strict=False)
        exp.load_state_dict(states["diffuser.expander"], strict=False)
        return diffuser, comp, exp

    def test_save_load_round_trip_keeps_stats(self, tmp_path):
        import torch

        diffuser, _, _, shift, scale = self._calibrated()
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.save_models(diffuser=diffuser, text_encoder=nn.Linear(2, 2))
        sd = safetensors.torch.load_file(str(mgr.model_path))
        for mod in ("compressor", "expander"):
            assert torch.equal(sd[f"diffuser.{mod}.latent_shift"], shift)
            assert torch.allclose(sd[f"diffuser.{mod}.latent_scale"], scale)
        diffuser2, comp2, exp2 = self._load_like_initialize_models(mgr)
        assert diffuser2.latent_stats_calibrated
        assert torch.equal(comp2.latent_shift, shift) and torch.equal(exp2.latent_shift, shift)
        assert torch.allclose(comp2.latent_scale, scale)

    def test_ema_applied_export_keeps_post_calibration_stats(self, tmp_path):
        """Orchestrator exports under ema.applied(); a pre-calibration EMA must not clobber."""
        import torch

        diffuser, comp, exp, flow, _ = self._models()
        ema = EMA(nn.ModuleList([comp, exp]), decay=0.999)  # shadow: identity stats
        shift, scale = torch.full((16,), 0.25), torch.full((16,), 1.5)
        diffuser.set_latent_stats(shift, scale)
        mgr = CheckpointManager(output_dir=tmp_path)
        with ema.applied():
            mgr.save_models(diffuser=diffuser, text_encoder=nn.Linear(2, 2))
        sd = safetensors.torch.load_file(str(mgr.model_path))
        assert torch.equal(sd["diffuser.compressor.latent_shift"], shift)
        assert torch.equal(sd["diffuser.expander.latent_scale"], scale)

    def test_training_state_resume_keeps_stats(self, tmp_path):
        import torch

        diffuser, comp, exp, shift, scale = self._calibrated()
        ema = EMA(nn.ModuleList([comp, exp]), decay=0.999)
        mgr = CheckpointManager(output_dir=tmp_path)
        mgr.save_training_state(
            epoch=0, batch_idx=1, global_step=1, optimizers={}, schedulers={}, ema=ema
        )
        _, comp2, exp2, _, _ = self._models()
        ema2 = EMA(nn.ModuleList([comp2, exp2]), decay=0.999)
        assert mgr.load_optimizer_scheduler_ema_states({}, {}, ema2)
        assert torch.equal(comp2.latent_shift, shift) and torch.equal(exp2.latent_shift, shift)
        with ema2.applied():
            assert torch.equal(comp2.latent_shift, shift)

    def test_save_models_to_explicit_path(self, tmp_path):
        """CLI export: full key layout written to an arbitrary file, no sibling files."""
        diffuser, _, _, shift, _ = self._calibrated()
        mgr = CheckpointManager(output_dir=tmp_path / "ckpt")
        out = tmp_path / "elsewhere" / "cal.safetensors"
        mgr.save_models(diffuser=diffuser, text_encoder=nn.Linear(2, 2), model_path=out)
        sd = safetensors.torch.load_file(str(out))
        assert "diffuser.compressor.latent_shift" in sd and "text_encoder.weight" in sd
        assert not (out.parent / "text_encoder.safetensors").exists()
        assert not mgr.model_path.exists()
