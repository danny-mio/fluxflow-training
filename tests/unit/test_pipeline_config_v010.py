"""Unit tests for v0.10.0 pipeline_config additions.

Covers the new keys per design doc §5.7 and the backward-compat aliases
for the legacy ``kl_beta`` / ``kl_warmup_steps`` field names.
"""

import warnings

from fluxflow_training.training.pipeline_config import (
    PipelineStepConfig,
    parse_pipeline_config,
)


def _wrap_step(step_dict: dict) -> dict:
    """Wrap a step dict into a valid top-level pipeline config dict."""
    return {"steps": [step_dict]}


class TestNewConfigDefaults:
    """v0.10.0 defaults per design doc §5.7."""

    def test_kl_z_weight_default(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.kl_z_weight == 0.5

    def test_kl_z_warmup_steps_default(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.kl_z_warmup_steps == 10000

    def test_ctx_shrinkage_weight_default(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.ctx_shrinkage_weight == 0.001

    def test_ctx_shrinkage_warmup_start_step_default(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.ctx_shrinkage_warmup_start_step == 5000

    def test_ctx_shrinkage_warmup_steps_default(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.ctx_shrinkage_warmup_steps == 5000

    def test_t_txt_default(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.t_txt == 32

    def test_null_prompt_default(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.null_prompt == ""


class TestNewConfigYAMLLoading:
    """Parser must accept the new keys."""

    def test_kl_z_weight_loads_from_yaml(self):
        cfg_dict = _wrap_step(
            {"name": "vae", "n_epochs": 1, "train_vae": True, "kl_z_weight": 0.25}
        )
        cfg = parse_pipeline_config(cfg_dict)
        assert cfg.steps[0].kl_z_weight == 0.25

    def test_ctx_shrinkage_keys_load_from_yaml(self):
        cfg_dict = _wrap_step(
            {
                "name": "vae",
                "n_epochs": 1,
                "train_vae": True,
                "ctx_shrinkage_weight": 0.005,
                "ctx_shrinkage_warmup_start_step": 1000,
                "ctx_shrinkage_warmup_steps": 2000,
            }
        )
        cfg = parse_pipeline_config(cfg_dict)
        assert cfg.steps[0].ctx_shrinkage_weight == 0.005
        assert cfg.steps[0].ctx_shrinkage_warmup_start_step == 1000
        assert cfg.steps[0].ctx_shrinkage_warmup_steps == 2000

    def test_t_txt_and_null_prompt_load_from_yaml(self):
        cfg_dict = _wrap_step(
            {
                "name": "flow",
                "n_epochs": 1,
                "train_diff": True,
                "t_txt": 64,
                "null_prompt": "an empty scene",
            }
        )
        cfg = parse_pipeline_config(cfg_dict)
        assert cfg.steps[0].t_txt == 64
        assert cfg.steps[0].null_prompt == "an empty scene"


class TestLegacyAliases:
    """Old keys must still work but emit DeprecationWarning."""

    def test_kl_beta_emits_deprecation_warning_and_still_loads(self):
        cfg_dict = _wrap_step({"name": "vae", "n_epochs": 1, "train_vae": True, "kl_beta": 0.123})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = parse_pipeline_config(cfg_dict)
        # Old field is preserved on the dataclass for backward compat
        assert cfg.steps[0].kl_beta == 0.123
        # ... and the new field is populated from the alias
        assert cfg.steps[0].kl_z_weight == 0.123
        # Deprecation warning fired
        assert any(
            issubclass(w.category, DeprecationWarning) and "kl_beta" in str(w.message)
            for w in caught
        )

    def test_kl_warmup_steps_emits_deprecation_warning_and_still_loads(self):
        cfg_dict = _wrap_step(
            {
                "name": "vae",
                "n_epochs": 1,
                "train_vae": True,
                "kl_warmup_steps": 7500,
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = parse_pipeline_config(cfg_dict)
        assert cfg.steps[0].kl_warmup_steps == 7500
        assert cfg.steps[0].kl_z_warmup_steps == 7500
        assert any(
            issubclass(w.category, DeprecationWarning) and "kl_warmup_steps" in str(w.message)
            for w in caught
        )

    def test_new_key_wins_over_legacy(self):
        cfg_dict = _wrap_step(
            {
                "name": "vae",
                "n_epochs": 1,
                "train_vae": True,
                "kl_beta": 0.999,
                "kl_z_weight": 0.1,
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = parse_pipeline_config(cfg_dict)
        # New key wins for the new field; legacy field carries the legacy value.
        assert cfg.steps[0].kl_z_weight == 0.1
        assert cfg.steps[0].kl_beta == 0.999

    def test_no_warning_when_only_new_keys(self):
        cfg_dict = _wrap_step(
            {
                "name": "vae",
                "n_epochs": 1,
                "train_vae": True,
                "kl_z_weight": 0.5,
                "kl_z_warmup_steps": 10000,
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parse_pipeline_config(cfg_dict)
        deprecations = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning)
            and ("kl_beta" in str(w.message) or "kl_warmup_steps" in str(w.message))
        ]
        assert deprecations == []


class TestDirectConstructionAcceptsNewKeys:
    """Direct PipelineStepConfig(...) must accept the new keys without TypeError."""

    def test_direct_construction_with_new_keys(self):
        step = PipelineStepConfig(
            name="s",
            n_epochs=1,
            train_vae=True,
            kl_z_weight=0.4,
            kl_z_warmup_steps=15000,
            ctx_shrinkage_weight=0.002,
            ctx_shrinkage_warmup_start_step=4000,
            ctx_shrinkage_warmup_steps=6000,
            t_txt=48,
            null_prompt="nothing",
        )
        assert step.kl_z_weight == 0.4
        assert step.kl_z_warmup_steps == 15000
        assert step.ctx_shrinkage_weight == 0.002
        assert step.ctx_shrinkage_warmup_start_step == 4000
        assert step.ctx_shrinkage_warmup_steps == 6000
        assert step.t_txt == 48
        assert step.null_prompt == "nothing"


class TestRandomLatentConfigDefaults:
    """Random-latent compressor training (independently gated VAE loss)."""

    def test_train_random_latent_default(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.train_random_latent is False

    def test_lambda_random_latent_default(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.lambda_random_latent == 1.0

    def test_random_latent_keys_load_from_yaml(self):
        cfg_dict = _wrap_step(
            {
                "name": "vae",
                "n_epochs": 1,
                "train_vae": True,
                "train_random_latent": True,
                "lambda_random_latent": 2.5,
            }
        )
        cfg = parse_pipeline_config(cfg_dict)
        assert cfg.steps[0].train_random_latent is True
        assert cfg.steps[0].lambda_random_latent == 2.5

    def test_random_latent_keys_load_from_yaml_default_when_absent(self):
        cfg_dict = _wrap_step({"name": "vae", "n_epochs": 1, "train_vae": True})
        cfg = parse_pipeline_config(cfg_dict)
        assert cfg.steps[0].train_random_latent is False
        assert cfg.steps[0].lambda_random_latent == 1.0

    def test_direct_construction_with_random_latent_keys(self):
        step = PipelineStepConfig(
            name="s",
            n_epochs=1,
            train_vae=True,
            train_random_latent=True,
            lambda_random_latent=3.0,
        )
        assert step.train_random_latent is True
        assert step.lambda_random_latent == 3.0
