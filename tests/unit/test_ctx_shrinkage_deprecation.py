"""ctx_shrinkage was removed (inert: acted before an affine-free instance norm).

Config keys stay parseable but are deprecated, ignored, and warned about once.
"""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from fluxflow.models.v100.vae import FluxCompressor_v100

from fluxflow_training.training import pipeline_config
from fluxflow_training.training.pipeline_config import parse_pipeline_config
from fluxflow_training.training.vae_trainer import VAETrainer

from tests.unit.test_scripts import import_script_module

_MSG = "ctx_shrinkage removed (inert: acted before an affine-free instance norm)"
_KEYS = (
    "ctx_shrinkage_weight",
    "ctx_shrinkage_warmup_start_step",
    "ctx_shrinkage_warmup_steps",
    "ctx_shrinkage_max_mean_sq",
)


@pytest.fixture(autouse=True)
def _reset_warn_flag(monkeypatch):
    monkeypatch.setattr(pipeline_config, "_CTX_SHRINKAGE_WARNED", False)


def _step(**extra) -> dict:
    return {"name": "vae", "n_epochs": 1, "train_vae": True, **extra}


def _count(caplog) -> int:
    return sum(_MSG in r.getMessage() for r in caplog.records)


class TestConfigDeprecation:
    def test_warns_once_and_drops_attribute(self, caplog):
        with caplog.at_level(logging.WARNING):
            cfg = parse_pipeline_config(
                {
                    "steps": [
                        _step(ctx_shrinkage_weight=0.001),
                        _step(name="vae2", ctx_shrinkage_warmup_steps=10),
                    ]
                }
            )
        assert _count(caplog) == 1
        for step in cfg.steps:
            for key in _KEYS:
                assert not hasattr(step, key)

    def test_all_legacy_keys_accepted(self, caplog):
        with caplog.at_level(logging.WARNING):
            parse_pipeline_config({"steps": [_step(**dict.fromkeys(_KEYS, 1))]})
        assert _count(caplog) == 1

    def test_no_warning_without_keys(self, caplog):
        with caplog.at_level(logging.WARNING):
            parse_pipeline_config({"steps": [_step()]})
        assert _count(caplog) == 0


class TestTrainerHasNoHook:
    def test_no_pre_hook_and_no_ctx_shrinkage_args(self):
        compressor = FluxCompressor_v100(d_model=8, downscales=2, use_gradient_checkpointing=False)
        expander = nn.Linear(1, 1)
        opt = torch.optim.SGD(expander.parameters(), lr=1e-3)
        trainer = VAETrainer(
            compressor=compressor,
            expander=expander,
            optimizer=opt,
            scheduler=torch.optim.lr_scheduler.StepLR(opt, step_size=1),
            ema=MagicMock(),
            reconstruction_loss_fn=nn.L1Loss(),
            reconstruction_loss_min_fn=nn.MSELoss(),
            train_reconstruction=False,
            train_kl=False,
            train_colorstats=False,
            train_histogram=False,
            train_contrast=False,
            train_coarseness=False,
            use_gan=False,
            use_lpips=False,
            ctx_input_dim=16,
            accelerator=None,
        )
        assert len(compressor.ctx_zinject_norm._forward_pre_hooks) == 0
        assert not hasattr(trainer, "_install_ctx_shrinkage_hook")
        with pytest.raises(TypeError):
            VAETrainer(ctx_shrinkage_weight=0.1)  # type: ignore[call-arg]


class TestLegacyCliFlags:
    def _parse(self, *extra):
        train = import_script_module("train")
        argv = ["train.py", "--data_path", "/tmp/i", "--captions_file", "/tmp/c.tsv", "--train_vae"]
        with patch.object(sys, "argv", argv + list(extra)):
            return train.parse_args()

    def test_flags_parse_warn_once_and_are_hidden(self, caplog):
        train = import_script_module("train")
        with caplog.at_level(logging.WARNING):
            self._parse("--ctx_shrinkage_weight", "0.01", "--ctx_shrinkage_max_mean_sq", "5")
        assert _count(caplog) == 1
        with patch.object(sys, "argv", ["train.py", "--help"]), pytest.raises(SystemExit):
            train.parse_args()

    def test_no_warning_when_unset(self, caplog):
        with caplog.at_level(logging.WARNING):
            self._parse()
        assert _count(caplog) == 0
