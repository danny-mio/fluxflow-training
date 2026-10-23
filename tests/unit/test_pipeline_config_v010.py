"""Unit tests for v0.10.0 pipeline_config additions.

Covers the new keys per design doc §5.7 and the backward-compat aliases
for the legacy ``kl_beta`` / ``kl_warmup_steps`` field names.
"""

import warnings

import pytest

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
        assert step.kl_z_weight == 1e-3

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

    def test_ctx_shrinkage_max_mean_sq_default(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.ctx_shrinkage_max_mean_sq == 1000.0

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
                "ctx_shrinkage_max_mean_sq": 500.0,
            }
        )
        cfg = parse_pipeline_config(cfg_dict)
        assert cfg.steps[0].ctx_shrinkage_weight == 0.005
        assert cfg.steps[0].ctx_shrinkage_warmup_start_step == 1000
        assert cfg.steps[0].ctx_shrinkage_warmup_steps == 2000
        assert cfg.steps[0].ctx_shrinkage_max_mean_sq == 500.0

    def test_ctx_shrinkage_max_mean_sq_defaults_when_omitted(self):
        cfg_dict = _wrap_step(
            {
                "name": "vae",
                "n_epochs": 1,
                "train_vae": True,
                "ctx_shrinkage_weight": 0.005,
            }
        )
        cfg = parse_pipeline_config(cfg_dict)
        assert cfg.steps[0].ctx_shrinkage_max_mean_sq == 1000.0

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
            ctx_shrinkage_max_mean_sq=250.0,
            t_txt=48,
            null_prompt="nothing",
        )
        assert step.kl_z_weight == 0.4
        assert step.kl_z_warmup_steps == 15000
        assert step.ctx_shrinkage_weight == 0.002
        assert step.ctx_shrinkage_warmup_start_step == 4000
        assert step.ctx_shrinkage_warmup_steps == 6000
        assert step.ctx_shrinkage_max_mean_sq == 250.0
        assert step.t_txt == 48
        assert step.null_prompt == "nothing"


_CYCLE_DEFAULTS = {
    "train_latent_cycle": False,
    "lambda_latent_cycle": 1.0,
    "latent_cycle_lpips_weight": 0.0,
    "latent_cycle_start_step": 0,
    "latent_cycle_warmup_steps": 0,
    "latent_cycle_every": 1,
    "latent_cycle_batch_frac": 1.0,
    "latent_cycle_crop_frac": 1.0,
    "latent_cycle_posterior_frac": 0.0,
    "latent_cycle_reject_range": 1.05,
    "latent_cycle_reject_frac": 0.01,
}
_CYCLE_CUSTOM = {
    "train_latent_cycle": True,
    "lambda_latent_cycle": 2.5,
    "latent_cycle_lpips_weight": 0.3,
    "latent_cycle_start_step": 100,
    "latent_cycle_warmup_steps": 50,
    "latent_cycle_every": 4,
    "latent_cycle_batch_frac": 0.5,
    "latent_cycle_crop_frac": 0.75,
    "latent_cycle_posterior_frac": 0.25,
    "latent_cycle_reject_range": 1.2,
    "latent_cycle_reject_frac": 0.05,
}


class TestLatentCycleConfig:
    """Latent cycle (image-space G.E idempotence) config."""

    def test_defaults(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        for key, val in _CYCLE_DEFAULTS.items():
            assert getattr(step, key) == val, key

    def test_yaml_defaults_when_absent(self):
        cfg = parse_pipeline_config(_wrap_step({"name": "v", "n_epochs": 1, "train_vae": True}))
        for key, val in _CYCLE_DEFAULTS.items():
            assert getattr(cfg.steps[0], key) == val, key

    def test_yaml_values_load(self):
        step_dict = {"name": "v", "n_epochs": 1, "train_vae": True, **_CYCLE_CUSTOM}
        cfg = parse_pipeline_config(_wrap_step(step_dict))
        for key, val in _CYCLE_CUSTOM.items():
            assert getattr(cfg.steps[0], key) == val, key

    def test_direct_construction(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True, **_CYCLE_CUSTOM)
        for key, val in _CYCLE_CUSTOM.items():
            assert getattr(step, key) == val, key

    @pytest.mark.parametrize(
        "key,bad",
        [
            ("lambda_latent_cycle", -0.1),
            ("latent_cycle_lpips_weight", -1.0),
            ("latent_cycle_start_step", -1),
            ("latent_cycle_warmup_steps", -1),
            ("latent_cycle_every", 0),
            ("latent_cycle_batch_frac", 0.0),
            ("latent_cycle_batch_frac", 1.5),
            ("latent_cycle_crop_frac", 0.0),
            ("latent_cycle_crop_frac", 1.01),
            ("latent_cycle_posterior_frac", -0.1),
            ("latent_cycle_posterior_frac", 1.1),
            ("latent_cycle_reject_range", 0.0),
            ("latent_cycle_reject_frac", -0.1),
            ("latent_cycle_reject_frac", 1.1),
        ],
    )
    def test_invalid_values_rejected(self, key, bad):
        step_dict = {"name": "v", "n_epochs": 1, "train_vae": True, key: bad}
        with pytest.raises(ValueError, match=key):
            parse_pipeline_config(_wrap_step(step_dict))

    @pytest.mark.parametrize(
        "key,ok",
        [
            ("latent_cycle_batch_frac", 1.0),
            ("latent_cycle_crop_frac", 0.01),
            ("latent_cycle_posterior_frac", 0.0),
            ("latent_cycle_posterior_frac", 1.0),
            ("latent_cycle_reject_frac", 0.0),
            ("latent_cycle_reject_frac", 1.0),
        ],
    )
    def test_boundary_values_accepted(self, key, ok):
        step_dict = {"name": "v", "n_epochs": 1, "train_vae": True, key: ok}
        assert getattr(parse_pipeline_config(_wrap_step(step_dict)).steps[0], key) == ok


class TestCtxAuxTrainingGateConfig:
    """train_ctx_aux gate for the ctx_aux_loss term (was hardcoded True)."""

    def test_train_ctx_aux_default(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.train_ctx_aux is False

    def test_train_ctx_aux_keys_load_from_yaml(self):
        cfg_dict = _wrap_step(
            {
                "name": "vae",
                "n_epochs": 1,
                "train_vae": True,
                "train_ctx_aux": False,
            }
        )
        cfg = parse_pipeline_config(cfg_dict)
        assert cfg.steps[0].train_ctx_aux is False

    def test_train_ctx_aux_keys_load_from_yaml_default_when_absent(self):
        cfg_dict = _wrap_step({"name": "vae", "n_epochs": 1, "train_vae": True})
        cfg = parse_pipeline_config(cfg_dict)
        assert cfg.steps[0].train_ctx_aux is False

    def test_direct_construction_with_train_ctx_aux_key(self):
        step = PipelineStepConfig(
            name="s",
            n_epochs=1,
            train_vae=True,
            train_ctx_aux=False,
        )
        assert step.train_ctx_aux is False


class TestDiscriminatorUseSpectralNormConfig:
    """Discriminator spectral norm (Miyato et al. 2018) toggle. Defaults to True
    (safer setting) -- deliberate behavior change for all existing configs."""

    def test_discriminator_use_spectral_norm_default(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.discriminator_use_spectral_norm is None  # inherits CLI (default True)

    def test_discriminator_use_spectral_norm_keys_load_from_yaml(self):
        cfg_dict = _wrap_step(
            {
                "name": "vae",
                "n_epochs": 1,
                "train_vae": True,
                "discriminator_use_spectral_norm": False,
            }
        )
        cfg = parse_pipeline_config(cfg_dict)
        assert cfg.steps[0].discriminator_use_spectral_norm is False

    def test_discriminator_use_spectral_norm_keys_load_from_yaml_default_when_absent(self):
        cfg_dict = _wrap_step({"name": "vae", "n_epochs": 1, "train_vae": True})
        cfg = parse_pipeline_config(cfg_dict)
        assert cfg.steps[0].discriminator_use_spectral_norm is None

    def test_direct_construction_with_discriminator_use_spectral_norm_key(self):
        step = PipelineStepConfig(
            name="s",
            n_epochs=1,
            train_vae=True,
            discriminator_use_spectral_norm=False,
        )
        assert step.discriminator_use_spectral_norm is False


class TestGanScheduleAndNoiseConfig:
    """gan_start_step / gan_warmup_steps / instance_noise_std defaults + validation."""

    def test_defaults(self):
        step = PipelineStepConfig(name="s", n_epochs=1, train_vae=True)
        assert step.gan_start_step == 0
        assert step.gan_warmup_steps == 1000
        assert step.instance_noise_std == 0.05
        assert step.discriminator_use_spectral_norm is None  # inherit CLI

    def test_yaml_defaults_and_overrides(self):
        base = {"name": "g", "n_epochs": 1, "gan_training": True, "train_vae": True}
        s = parse_pipeline_config(_wrap_step(base)).steps[0]
        assert (s.gan_start_step, s.gan_warmup_steps, s.instance_noise_std) == (0, 1000, 0.05)
        s = parse_pipeline_config(
            _wrap_step({**base, "gan_start_step": 7, "gan_warmup_steps": 9})
        ).steps[0]
        assert (s.gan_start_step, s.gan_warmup_steps) == (7, 9)

    @pytest.mark.parametrize(
        "key,val",
        [
            ("gan_start_step", -1),
            ("gan_warmup_steps", -1),
            ("kl_z_warmup_steps", -1),
            ("instance_noise_std", -0.1),
        ],
    )
    def test_negative_rejected(self, key, val):
        step = {"name": "g", "n_epochs": 1, "train_vae": True, key: val}
        with pytest.raises(ValueError, match=key):
            parse_pipeline_config(_wrap_step(step))

    def test_legacy_kl_warmup_steps_remaps_to_kl_z_warmup_steps(self):
        step = {"name": "g", "n_epochs": 1, "train_vae": True, "kl_warmup_steps": 123}
        with pytest.warns(DeprecationWarning):
            cfg = parse_pipeline_config(_wrap_step(step))
        assert cfg.steps[0].kl_z_warmup_steps == 123
