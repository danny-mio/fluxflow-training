"""scripts/train.py audit fixes: legacy VAETrainer wiring (A2), new CLI flags, KL schedule
reporting, D build/load compat (C5/C6), scheduler budgets (A3), EMA exports (D3)."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from fluxflow.models import PatchDiscriminator

from tests.unit.test_scripts import import_script_module

SRC = (
    Path(__file__).parent.parent.parent / "src" / "fluxflow_training" / "scripts" / "train.py"
).read_text()


def _parse(*extra, config=None):
    train = import_script_module("train")
    argv = ["train.py", "--data_path", "/tmp/i", "--captions_file", "/tmp/c.tsv", "--train_vae"]
    argv += list(extra)
    if config:
        argv += ["--config", str(config)]
    with patch.object(sys, "argv", argv):
        return train.parse_args()


class TestNewCliFlags:
    def test_defaults(self):
        a = _parse()
        assert a.kl_z_warmup_steps == 10000
        assert a.ctx_shrinkage_warmup_start_step == 5000
        assert a.ctx_shrinkage_warmup_steps == 5000
        assert a.ctx_shrinkage_max_mean_sq == 1000.0
        assert a.gan_start_step == 0
        assert a.gan_warmup_steps == 1000
        assert a.instance_noise_std == 0.05
        assert a.instance_noise_decay == 0.9999
        assert a.train_ctx_aux is False
        assert a.lambda_ctx_aux == 0.01

    def test_cli_overrides(self):
        a = _parse(
            "--gan_start_step", "7", "--gan_warmup_steps", "9", "--kl_z_warmup_steps", "3",
            "--instance_noise_std", "0.2", "--train_ctx_aux",
        )  # fmt: skip
        assert (a.gan_start_step, a.gan_warmup_steps, a.kl_z_warmup_steps) == (7, 9, 3)
        assert a.instance_noise_std == 0.2 and a.train_ctx_aux is True

    def test_yaml_overrides_and_cli_wins(self, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "training:\n  gan_start_step: 11\n  gan_warmup_steps: 12\n"
            "  kl_z_warmup_steps: 13\n  train_ctx_aux: true\n  instance_noise_std: 0.3\n"
        )
        a = _parse("--gan_warmup_steps", "99", config=cfg)
        assert a.gan_start_step == 11 and a.gan_warmup_steps == 99
        assert a.kl_z_warmup_steps == 13 and a.train_ctx_aux is True
        assert a.instance_noise_std == 0.3


class TestLegacyVaeTrainerWiring:
    @pytest.mark.parametrize(
        "kw",
        [
            "kl_z_warmup_steps=args.kl_z_warmup_steps",
            "ctx_shrinkage_warmup_start_step=args.ctx_shrinkage_warmup_start_step",
            "ctx_shrinkage_warmup_steps=args.ctx_shrinkage_warmup_steps",
            "ctx_shrinkage_max_mean_sq=args.ctx_shrinkage_max_mean_sq",
            "gan_start_step=args.gan_start_step",
            "gan_warmup_steps=args.gan_warmup_steps",
            "instance_noise_std=args.instance_noise_std",
            "instance_noise_decay=args.instance_noise_decay",
            "train_ctx_aux=args.train_ctx_aux",
            "lambda_ctx_aux=args.lambda_ctx_aux",
            "total_training_steps=total_global_steps",
        ],
    )
    def test_kwarg_passed(self, kw):
        assert kw in SRC


class TestKlSchedule:
    def test_legacy_when_kl_z_off(self):
        from fluxflow_training.scripts import train

        a = SimpleNamespace(kl_z_weight=0.0, kl_z_warmup_steps=10, kl_beta=0.1, kl_warmup_steps=20)
        assert train._kl_schedule(a) == (20, 0.1)

    def test_kl_z_when_active(self):
        from fluxflow_training.scripts import train

        a = SimpleNamespace(kl_z_weight=0.5, kl_z_warmup_steps=10, kl_beta=0.1, kl_warmup_steps=20)
        assert train._kl_schedule(a) == (10, 0.5)

    def test_no_raw_legacy_kl_args_left_in_logging_or_metadata(self):
        assert "cosine_anneal_beta(global_step, args.kl_warmup_steps" not in SRC
        assert "kl_max_beta=args.kl_beta" not in SRC


class TestMakeDiscriminator:
    def _args(self, sn=True):
        return SimpleNamespace(
            channels=3, feature_maps_dim_disc=4, discriminator_use_spectral_norm=sn
        )

    def test_loads_matching_state(self):
        from fluxflow_training.scripts import train

        saved = PatchDiscriminator(3, 4, 3, ctx_dim=0, use_spectral_norm=True)
        d = train._make_discriminator(self._args(True), 0, saved.state_dict())
        assert torch.equal(d.head.weight_orig, saved.head.weight_orig)

    def test_spectral_norm_mismatch_reinits_without_claiming_success(self, capsys):
        from fluxflow_training.scripts import train

        saved = PatchDiscriminator(3, 4, 3, ctx_dim=0, use_spectral_norm=False)
        d = train._make_discriminator(self._args(True), 0, saved.state_dict())
        out = capsys.readouterr().out
        assert "Loaded D_img" not in out and "Reinitialising" in out
        assert has_sn(d)

    def test_legacy_ctx_proj_ignored_for_v100(self, capsys):
        from fluxflow_training.scripts import train

        saved = PatchDiscriminator(3, 4, 3, ctx_dim=16, use_spectral_norm=True)
        d = train._make_discriminator(self._args(True), 0, saved.state_dict())
        assert "Loaded D_img" in capsys.readouterr().out
        assert torch.equal(d.head.weight_orig, saved.head.weight_orig)
        assert not hasattr(d, "ctx_proj")

    def test_nan_state_reinitialised(self, capsys):
        from fluxflow_training.scripts import train

        saved = PatchDiscriminator(3, 4, 3, ctx_dim=0, use_spectral_norm=True)
        with torch.no_grad():
            saved.head.weight_orig.fill_(float("nan"))
        d = train._make_discriminator(self._args(True), 0, saved.state_dict())
        assert not torch.isnan(d.head.weight_orig).any()


def has_sn(m):
    from fluxflow_training.training.utils import has_spectral_norm

    return has_spectral_norm(m)


class TestLegacySchedulerBudgetAndEma:
    def test_vae_and_disc_schedulers_use_global_step_budget(self):
        assert (
            'create_scheduler(optimizer_Diff, scheduler_configs["vae"], total_global_steps)' in SRC
        )
        assert 'scheduler_configs["discriminator"], total_global_steps' in SRC

    def test_exports_and_samples_use_ema(self):
        assert "ema.applied()" in SRC
