"""ctx_dropout_frac / ctx_noise_sigma wiring: pipeline config, orchestrator, legacy CLI."""

from pathlib import Path

import pytest

from fluxflow_training.training.pipeline_config import PipelineStepConfig, parse_pipeline_config
from tests.unit.test_pipeline_config_v010 import _wrap_step

_ROOT = Path(__file__).parent.parent.parent
_SRC = _ROOT / "src" / "fluxflow_training"
_TRAIN = (_SRC / "scripts" / "train.py").read_text()
_ORCH = (_SRC / "training" / "pipeline_orchestrator.py").read_text()


def _parse(**kw):
    return parse_pipeline_config(_wrap_step({"name": "v", "n_epochs": 1, "train_vae": True, **kw}))


class TestPipelineConfig:
    def test_defaults(self):
        s = PipelineStepConfig(name="s", n_epochs=1)
        assert (s.ctx_dropout_frac, s.ctx_noise_sigma) == (0.5, 0.5)
        assert (_parse().steps[0].ctx_dropout_frac, _parse().steps[0].ctx_noise_sigma) == (0.5, 0.5)

    def test_yaml_values_load(self):
        s = _parse(ctx_dropout_frac=0.25, ctx_noise_sigma=0.0).steps[0]
        assert (s.ctx_dropout_frac, s.ctx_noise_sigma) == (0.25, 0.0)

    @pytest.mark.parametrize(
        "key,bad",
        [("ctx_dropout_frac", -0.1), ("ctx_dropout_frac", 1.0), ("ctx_noise_sigma", -0.1)],
    )
    def test_invalid_rejected(self, key, bad):
        with pytest.raises(ValueError, match=key):
            _parse(**{key: bad})

    @pytest.mark.parametrize(
        "key,ok",
        [
            ("ctx_dropout_frac", 0.0),
            ("ctx_dropout_frac", 0.99),
            ("ctx_noise_sigma", 0.0),
            ("ctx_noise_sigma", 3.0),
        ],
    )
    def test_valid_accepted(self, key, ok):
        assert getattr(_parse(**{key: ok}).steps[0], key) == ok


class TestOrchestrator:
    @pytest.mark.parametrize("key", ["ctx_dropout_frac", "ctx_noise_sigma"])
    def test_passed_to_vae_trainer(self, key):
        assert f'{key}=getattr(step, "{key}", 0.5)' in _ORCH

    def test_logs_diagnostics(self):
        assert "L1ctx" in _ORCH and "L1z" in _ORCH
        assert 'metrics["recon_l1_ctx"]' in _ORCH and 'metrics["recon_l1_zonly"]' in _ORCH


class TestLegacyCli:
    @pytest.mark.parametrize("key", ["ctx_dropout_frac", "ctx_noise_sigma"])
    def test_arg_yaml_and_kwarg(self, key):
        assert f'"--{key}"' in _TRAIN
        assert f'"{key}",' in _TRAIN.split("_YAML_TRAINING_KEYS = (")[1].split(")")[0]
        assert f"{key}=args.{key}" in _TRAIN

    def test_defaults_and_cli_wins(self):
        from fluxflow_training.scripts import train

        assert "ctx_dropout_frac" in train._YAML_TRAINING_KEYS


class TestDocs:
    @pytest.mark.parametrize("f", ["config.example.yaml", "config.cfg.example.yaml"])
    def test_example_configs(self, f):
        t = (_ROOT / f).read_text()
        assert "ctx_dropout_frac" in t and "ctx_noise_sigma" in t

    def test_changelog(self):
        assert "ctx_dropout_frac" in (_ROOT / "CHANGELOG.md").read_text()


class TestLegacyParse:
    def _parse(self, *extra, config=None):

        from tests.unit.test_train_script_audit_fixes import _parse

        return _parse(*extra, config=config)

    def test_defaults(self):
        a = self._parse()
        assert (a.ctx_dropout_frac, a.ctx_noise_sigma) == (0.5, 0.5)

    def test_yaml_and_cli_wins(self, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("training:\n  ctx_dropout_frac: 0.25\n  ctx_noise_sigma: 0.1\n")
        a = self._parse(config=cfg)
        assert (a.ctx_dropout_frac, a.ctx_noise_sigma) == (0.25, 0.1)
        a = self._parse("--ctx_noise_sigma", "0", config=cfg)
        assert (a.ctx_dropout_frac, a.ctx_noise_sigma) == (0.25, 0.0)
