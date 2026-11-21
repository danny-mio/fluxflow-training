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
