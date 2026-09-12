"""Tests for CheckpointManager.save_models metadata whitelist.

Covers the activation_type (Bezier vs Padé) metadata fields specifically --
these must round-trip through the saved checkpoint's safetensors header so
other tools (fluxflow-ui, fluxflow-comfyui) can read back which activation
family a checkpoint was trained with.
"""

import safetensors.torch
import torch.nn as nn

from fluxflow_training.training.checkpoint_manager import CheckpointManager


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
