"""Unit tests for optimizer factory (fluxflow_training/training/optimizer_factory.py)."""

import pytest
import torch
import torch.nn as nn

from fluxflow_training.training.optimizer_factory import (
    SUPPORTED_OPTIMIZERS,
    create_optimizer,
    get_default_optimizer_config,
    validate_optimizer_config,
)


class TestValidateOptimizerConfig:
    """Tests for validate_optimizer_config function."""

    def test_valid_adamw_config(self):
        """Valid AdamW config should pass validation."""
        config = {
            "type": "AdamW",
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
        }
        validate_optimizer_config(config)  # Should not raise

    def test_valid_lion_config(self):
        """Valid Lion config should pass validation."""
        config = {
            "type": "Lion",
            "lr": 5e-7,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
        }
        validate_optimizer_config(config)  # Should not raise

    def test_valid_sgd_config(self):
        """Valid SGD config should pass validation."""
        config = {
            "type": "SGD",
            "lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0001,
        }
        validate_optimizer_config(config)  # Should not raise

    def test_valid_rmsprop_config(self):
        """Valid RMSprop config should pass validation."""
        config = {
            "type": "RMSprop",
            "lr": 0.001,
            "alpha": 0.99,
            "momentum": 0.0,
        }
        validate_optimizer_config(config)  # Should not raise

    def test_discriminator_default_config(self):
        """Discriminator default config (beta1=0.0) should pass validation."""
        config = {
            "type": "AdamW",
            "lr": 5e-7,
            "betas": [0.0, 0.9],
            "weight_decay": 0.001,
        }
        validate_optimizer_config(config)  # Should not raise

    def test_negative_lr_raises(self):
        """Negative learning rate should raise ValueError."""
        config = {"type": "AdamW", "lr": -0.001}
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            validate_optimizer_config(config)

    def test_zero_lr_raises(self):
        """Zero learning rate should raise ValueError."""
        config = {"type": "AdamW", "lr": 0}
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            validate_optimizer_config(config)

    def test_beta_equals_one_raises(self):
        """Beta value equal to 1.0 should raise ValueError."""
        config = {"type": "AdamW", "lr": 0.001, "betas": [1.0, 0.999]}
        with pytest.raises(ValueError, match="Beta1 must be in"):
            validate_optimizer_config(config)

    def test_beta_greater_than_one_raises(self):
        """Beta value greater than 1.0 should raise ValueError."""
        config = {"type": "AdamW", "lr": 0.001, "betas": [0.9, 1.5]}
        with pytest.raises(ValueError, match="Beta2 must be in"):
            validate_optimizer_config(config)

    def test_negative_beta_raises(self):
        """Negative beta value should raise ValueError."""
        config = {"type": "AdamW", "lr": 0.001, "betas": [-0.1, 0.999]}
        with pytest.raises(ValueError, match="Beta1 must be in"):
            validate_optimizer_config(config)

    def test_wrong_number_of_betas_raises(self):
        """Wrong number of beta values should raise ValueError."""
        config = {"type": "AdamW", "lr": 0.001, "betas": [0.9]}
        with pytest.raises(ValueError, match="Betas must have exactly 2 values"):
            validate_optimizer_config(config)

    def test_negative_weight_decay_raises(self):
        """Negative weight decay should raise ValueError."""
        config = {"type": "AdamW", "lr": 0.001, "weight_decay": -0.1}
        with pytest.raises(ValueError, match="Weight decay must be non-negative"):
            validate_optimizer_config(config)

    def test_zero_eps_raises(self):
        """Zero epsilon should raise ValueError."""
        config = {"type": "AdamW", "lr": 0.001, "eps": 0}
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            validate_optimizer_config(config)

    def test_negative_eps_raises(self):
        """Negative epsilon should raise ValueError."""
        config = {"type": "AdamW", "lr": 0.001, "eps": -1e-8}
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            validate_optimizer_config(config)

    def test_momentum_equals_one_raises(self):
        """Momentum equal to 1.0 should raise ValueError."""
        config = {"type": "SGD", "lr": 0.01, "momentum": 1.0}
        with pytest.raises(ValueError, match="Momentum must be in"):
            validate_optimizer_config(config)

    def test_negative_momentum_raises(self):
        """Negative momentum should raise ValueError."""
        config = {"type": "SGD", "lr": 0.01, "momentum": -0.1}
        with pytest.raises(ValueError, match="Momentum must be in"):
            validate_optimizer_config(config)

    def test_zero_alpha_raises(self):
        """Zero alpha for RMSprop should raise ValueError."""
        config = {"type": "RMSprop", "lr": 0.001, "alpha": 0}
        with pytest.raises(ValueError, match="Alpha must be in"):
            validate_optimizer_config(config)

    def test_alpha_greater_than_one_raises(self):
        """Alpha greater than 1.0 for RMSprop should raise ValueError."""
        config = {"type": "RMSprop", "lr": 0.001, "alpha": 1.5}
        with pytest.raises(ValueError, match="Alpha must be in"):
            validate_optimizer_config(config)


class TestCreateOptimizer:
    """Tests for create_optimizer function."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 5)

    def test_create_adamw(self, simple_model):
        """Should create AdamW optimizer."""
        config = {"type": "AdamW", "lr": 1e-4}
        optimizer = create_optimizer(simple_model.parameters(), config)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 1e-4

    def test_create_adam(self, simple_model):
        """Should create Adam optimizer."""
        config = {"type": "Adam", "lr": 1e-3}
        optimizer = create_optimizer(simple_model.parameters(), config)
        assert isinstance(optimizer, torch.optim.Adam)

    def test_create_sgd(self, simple_model):
        """Should create SGD optimizer."""
        config = {"type": "SGD", "lr": 0.01, "momentum": 0.9}
        optimizer = create_optimizer(simple_model.parameters(), config)
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["momentum"] == 0.9

    def test_create_rmsprop(self, simple_model):
        """Should create RMSprop optimizer."""
        config = {"type": "RMSprop", "lr": 0.001}
        optimizer = create_optimizer(simple_model.parameters(), config)
        assert isinstance(optimizer, torch.optim.RMSprop)

    def test_create_lion(self, simple_model):
        """Should create Lion optimizer."""
        config = {"type": "Lion", "lr": 5e-7}
        optimizer = create_optimizer(simple_model.parameters(), config)
        # Lion is from lion_pytorch package
        assert optimizer.__class__.__name__ == "Lion"

    def test_unsupported_optimizer_raises(self, simple_model):
        """Unsupported optimizer type should raise ValueError."""
        config = {"type": "InvalidOptimizer", "lr": 0.001}
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            create_optimizer(simple_model.parameters(), config)

    def test_betas_passed_correctly(self, simple_model):
        """Beta values should be passed to optimizer."""
        config = {"type": "AdamW", "lr": 1e-4, "betas": [0.85, 0.95]}
        optimizer = create_optimizer(simple_model.parameters(), config)
        assert optimizer.param_groups[0]["betas"] == (0.85, 0.95)

    def test_weight_decay_passed_correctly(self, simple_model):
        """Weight decay should be passed to optimizer."""
        config = {"type": "AdamW", "lr": 1e-4, "weight_decay": 0.05}
        optimizer = create_optimizer(simple_model.parameters(), config)
        assert optimizer.param_groups[0]["weight_decay"] == 0.05

    def test_default_lr_used(self, simple_model):
        """Default learning rate should be used if not specified."""
        config = {"type": "AdamW"}
        optimizer = create_optimizer(simple_model.parameters(), config)
        assert optimizer.param_groups[0]["lr"] == 1e-4


class TestGetDefaultOptimizerConfig:
    """Tests for get_default_optimizer_config function."""

    def test_flow_default(self):
        """Flow model should use Lion optimizer.

        lr=1e-7 (5x below AdamW's 5e-7) and weight_decay=0.05 (5x AdamW's 0.01)
        per Lion's published guidance (Chen et al. 2023): Lion's sign-based
        update is constant-magnitude per step, so it needs a smaller lr and
        larger weight_decay than an equivalent AdamW setting to converge
        instead of oscillating around a fixed point.
        """
        config = get_default_optimizer_config("flow")
        assert config["type"] == "Lion"
        assert config["lr"] == 1e-7
        assert config["weight_decay"] == 0.05

    def test_vae_default(self):
        """VAE model should use AdamW optimizer."""
        config = get_default_optimizer_config("vae")
        assert config["type"] == "AdamW"
        assert config["lr"] == 5e-7

    def test_text_encoder_default(self):
        """Text encoder should use AdamW with lower LR."""
        config = get_default_optimizer_config("text_encoder")
        assert config["type"] == "AdamW"
        assert config["lr"] == 5e-8  # 1/10 of flow

    def test_discriminator_default(self):
        """Discriminator should use AdamW with amsgrad."""
        config = get_default_optimizer_config("discriminator")
        assert config["type"] == "AdamW"
        assert config["amsgrad"] is True
        assert config["betas"] == [0.0, 0.9]

    def test_unknown_model_uses_vae_default(self):
        """Unknown model name should use VAE defaults."""
        config = get_default_optimizer_config("unknown_model")
        assert config["type"] == "AdamW"
        assert config["lr"] == 5e-7

    def test_all_defaults_pass_validation(self):
        """All default configs should pass validation."""
        for model_name in ["flow", "vae", "text_encoder", "discriminator"]:
            config = get_default_optimizer_config(model_name)
            validate_optimizer_config(config)  # Should not raise


class TestGetDefaultOptimizerConfigLionOverride:
    """Tests for the optional Lion-rescaled override (additive; vae/discriminator/
    text_encoder still default to AdamW unless optimizer_type='Lion' is requested).
    """

    def test_vae_default_unchanged_without_optimizer_type(self):
        """Omitting optimizer_type must not change vae's actual default (AdamW)."""
        config = get_default_optimizer_config("vae")
        assert config["type"] == "AdamW"
        assert config["lr"] == 5e-7
        assert config["weight_decay"] == 0.01

    def test_vae_lion_override(self):
        """vae + optimizer_type='Lion' should return Lion-appropriate values,
        not AdamW's lr=5e-7/weight_decay=0.01 reused verbatim.
        """
        config = get_default_optimizer_config("vae", optimizer_type="Lion")
        assert config["type"] == "Lion"
        assert config["lr"] == 1e-7  # 5x below AdamW's 5e-7
        assert config["weight_decay"] == 0.05  # 5x AdamW's 0.01

    def test_text_encoder_lion_override(self):
        """text_encoder + optimizer_type='Lion' should rescale relative to its
        own (already lower) AdamW lr, not vae's.
        """
        config = get_default_optimizer_config("text_encoder", optimizer_type="Lion")
        assert config["type"] == "Lion"
        assert config["lr"] == 1e-8  # 5x below AdamW's 5e-8
        assert config["weight_decay"] == 0.05  # 5x AdamW's 0.01

    def test_discriminator_lion_override(self):
        """discriminator + optimizer_type='Lion' should rescale relative to its
        own (already low) AdamW weight_decay, not vae's/flow's.
        """
        config = get_default_optimizer_config("discriminator", optimizer_type="Lion")
        assert config["type"] == "Lion"
        assert config["lr"] == 1e-7  # 5x below AdamW's 5e-7
        assert config["weight_decay"] == 0.005  # 5x AdamW's 0.001

    def test_flow_lion_override_is_a_no_op(self):
        """flow already defaults to Lion — requesting optimizer_type='Lion' must
        not double-apply the rescale.
        """
        config = get_default_optimizer_config("flow", optimizer_type="Lion")
        assert config["type"] == "Lion"
        assert config["lr"] == 1e-7
        assert config["weight_decay"] == 0.05

    def test_unknown_model_lion_override_falls_back_to_vae(self):
        """Unknown model + optimizer_type='Lion' should use vae's Lion override."""
        config = get_default_optimizer_config("unknown_model", optimizer_type="Lion")
        assert config["type"] == "Lion"
        assert config["lr"] == 1e-7
        assert config["weight_decay"] == 0.05

    def test_non_lion_optimizer_type_is_ignored(self):
        """A non-Lion optimizer_type does not trigger any rescaling — vae still
        returns its native AdamW default (only the Lion mismatch is addressed).
        """
        config = get_default_optimizer_config("vae", optimizer_type="SGD")
        assert config["type"] == "AdamW"
        assert config["lr"] == 5e-7

    def test_all_lion_overrides_pass_validation(self):
        """Every Lion-override config must itself be a valid optimizer config."""
        for model_name in ["vae", "text_encoder", "discriminator", "flow", "unknown_model"]:
            config = get_default_optimizer_config(model_name, optimizer_type="Lion")
            validate_optimizer_config(config)  # Should not raise


class TestSupportedOptimizers:
    """Tests for SUPPORTED_OPTIMIZERS constant."""

    def test_contains_expected_optimizers(self):
        """Should contain all expected optimizer types."""
        expected = ["Lion", "AdamW", "Adam", "SGD", "RMSprop"]
        for opt_name in expected:
            assert opt_name in SUPPORTED_OPTIMIZERS

    def test_optimizer_classes_are_callable(self):
        """All optimizer classes should be callable."""
        for opt_name, opt_class in SUPPORTED_OPTIMIZERS.items():
            assert callable(opt_class), f"{opt_name} is not callable"


class _Gated(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.conv1_scale = nn.Parameter(torch.zeros(1))
        self.spade = nn.Module()
        self.spade.beta_scale = nn.Parameter(torch.zeros(1))
        self.vec_scale = nn.Parameter(torch.zeros(3))  # numel != 1 -> decay
        self.other = nn.Parameter(torch.zeros(1))  # name mismatch -> decay


class TestWeightDecayGroups:
    def test_split_groups(self):
        from fluxflow_training.training.optimizer_factory import split_weight_decay_groups

        m = _Gated()
        decay, no_decay = split_weight_decay_groups(m.named_parameters())
        names = dict(m.named_parameters())
        assert {id(p) for p in no_decay["params"]} == {
            id(names["conv1_scale"]),
            id(names["spade.beta_scale"]),
        }
        assert no_decay["weight_decay"] == 0.0
        assert "weight_decay" not in decay
        assert len(decay["params"]) == 4

    def test_no_gating_params_gives_single_group(self):
        from fluxflow_training.training.optimizer_factory import split_weight_decay_groups

        groups = split_weight_decay_groups(nn.Linear(2, 2).named_parameters())
        assert len(groups) == 1

    @pytest.mark.parametrize("typ", ["AdamW", "Lion"])
    def test_create_optimizer_applies_group_decay(self, typ):
        from fluxflow_training.training.optimizer_factory import split_weight_decay_groups

        m = _Gated()
        opt = create_optimizer(
            split_weight_decay_groups(m.named_parameters()),
            {"type": typ, "lr": 1e-3, "weight_decay": 0.1},
        )
        assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]
        # zero gating scalar stays untouched by decay when grad is zero
        m.conv1_scale.data.fill_(1.0)
        for p in m.parameters():
            p.grad = torch.zeros_like(p)
        opt.step()
        if typ == "AdamW":
            assert m.conv1_scale.item() == 1.0
            assert m.other.item() == 0.0

    def test_scheduler_updates_every_group(self):
        from fluxflow_training.training.optimizer_factory import split_weight_decay_groups

        m = _Gated()
        opt = create_optimizer(
            split_weight_decay_groups(m.named_parameters()), {"type": "AdamW", "lr": 1.0}
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
        for _ in range(2):
            opt.step()
            sched.step()
        assert opt.param_groups[0]["lr"] == opt.param_groups[1]["lr"] < 1.0


class TestResumeOldSingleGroupState:
    def _old_opt_state(self, m):
        old = torch.optim.AdamW(list(m.parameters()), lr=0.5, weight_decay=0.1)
        for p in old.param_groups[0]["params"]:
            p.grad = torch.full_like(p, float(id(p) % 7 + 1))
        old.step()
        return old, old.state_dict()

    def test_remap_preserves_per_param_state(self):
        from fluxflow_training.training.optimizer_factory import (
            remap_optimizer_state,
            split_weight_decay_groups,
        )

        m = _Gated()
        old, saved = self._old_opt_state(m)
        opt = create_optimizer(
            split_weight_decay_groups(m.named_parameters()),
            {"type": "AdamW", "lr": 1e-3, "weight_decay": 0.1},
        )
        remapped = remap_optimizer_state(opt, saved)
        assert remapped is not None
        opt.load_state_dict(remapped)
        for p in m.parameters():
            assert torch.equal(opt.state[p]["exp_avg"], old.state[p]["exp_avg"])
        assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]
        assert opt.param_groups[0]["lr"] == 0.5

    def test_matching_group_count_unchanged(self):
        from fluxflow_training.training.optimizer_factory import remap_optimizer_state

        m = nn.Linear(2, 2)
        opt = torch.optim.AdamW(m.parameters())
        saved = opt.state_dict()
        assert remap_optimizer_state(opt, saved) is saved

    def test_unmappable_returns_none(self, caplog):
        from fluxflow_training.training.optimizer_factory import remap_optimizer_state

        m = _Gated()
        opt = torch.optim.AdamW([{"params": [m.lin.weight]}, {"params": [m.lin.bias]}])
        saved = torch.optim.AdamW(m.parameters()).state_dict()
        assert remap_optimizer_state(opt, saved) is None

    def test_checkpoint_manager_resume(self, tmp_path):
        from fluxflow_training.training.checkpoint_manager import CheckpointManager
        from fluxflow_training.training.optimizer_factory import split_weight_decay_groups

        m = _Gated()
        old, saved = self._old_opt_state(m)
        sched_old = torch.optim.lr_scheduler.CosineAnnealingLR(old, T_max=10)
        old.step()
        sched_old.step()
        mgr = CheckpointManager(output_dir=tmp_path)
        torch.save(
            {
                "optimizers": {"vae": old.state_dict()},
                "schedulers": {"vae": sched_old.state_dict()},
            },
            mgr.training_states_path,
        )
        # new-format optimizer groups contain orig_idx list; torch.save handles it
        opt = create_optimizer(
            split_weight_decay_groups(m.named_parameters()),
            {"type": "AdamW", "lr": 1.0, "weight_decay": 0.1},
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        try:
            mgr.load_optimizer_scheduler_ema_states({"vae": opt}, {"vae": sched}, _NoEma())
        finally:
            pass
        assert torch.equal(opt.state[m.lin.weight]["exp_avg"], old.state[m.lin.weight]["exp_avg"])
        assert len(sched.base_lrs) == 2 and len(sched._last_lr) == 2
        sched.step()
        assert opt.param_groups[0]["lr"] == opt.param_groups[1]["lr"]


class _NoEma:
    def load_state_dict(self, _):
        raise AssertionError("no ema in checkpoint")


class _ActNet(nn.Module):
    def __init__(self):
        super().__init__()
        from fluxflow.models import TrainableBezier, WideTrainableBezier
        from fluxflow.models.pade_activation import TrainablePade, WideTrainablePade

        self.lin = nn.Linear(2, 2)
        self.bez = TrainableBezier((2,), channel_only=True)
        self.wbez = WideTrainableBezier((2,), channel_only=True)
        self.pade = TrainablePade((2,), channel_only=True)
        self.wpade = WideTrainablePade((2,), channel_only=True)


class TestActivationWeightDecayExemption:
    """E7: activation coefficients must not be weight-decayed (detected by module type)."""

    def test_activation_params_in_no_decay_group(self):
        from fluxflow_training.training.optimizer_factory import split_weight_decay_groups

        m = _ActNet()
        groups = split_weight_decay_groups(m.named_parameters(), modules=[m])
        decay = [g for g in groups if "weight_decay" not in g]
        no_decay = [g for g in groups if g.get("weight_decay") == 0.0]
        assert len(decay) == 1 and len(no_decay) == 1
        assert {id(p) for p in decay[0]["params"]} == {id(m.lin.weight), id(m.lin.bias)}
        act_ids = {id(p) for sub in (m.bez, m.wbez, m.pade, m.wpade) for p in sub.parameters()}
        assert {id(p) for p in no_decay[0]["params"]} == act_ids

    def test_orig_idx_preserved_for_resume(self):
        from fluxflow_training.training.optimizer_factory import split_weight_decay_groups

        m = _ActNet()
        named = list(m.named_parameters())
        groups = split_weight_decay_groups(iter(named), modules=[m])
        idx = sorted(i for g in groups for i in g["orig_idx"])
        assert idx == list(range(len(named)))
        for g in groups:
            assert [named[i][1] for i in g["orig_idx"]] == g["params"]

    def test_without_modules_behaviour_unchanged(self):
        from fluxflow_training.training.optimizer_factory import split_weight_decay_groups

        m = _ActNet()
        groups = split_weight_decay_groups(m.named_parameters())
        assert len(groups) == 1 and "weight_decay" not in groups[0]

    def test_scale_scalars_still_exempt_with_modules(self):
        from fluxflow_training.training.optimizer_factory import split_weight_decay_groups

        m = _Gated()
        groups = split_weight_decay_groups(m.named_parameters(), modules=[m])
        nd = [g for g in groups if g.get("weight_decay") == 0.0][0]
        assert {id(p) for p in nd["params"]} == {id(m.conv1_scale), id(m.spade.beta_scale)}


class TestDiscriminatorDefaults:
    """C7: GAN-tuned discriminator defaults, not the VAE optimizer config."""

    def test_default_discriminator_config_is_gan_tuned(self):
        from fluxflow_training.training.optimizer_factory import get_default_optimizer_config

        cfg = get_default_optimizer_config("discriminator")
        assert cfg["betas"] == [0.0, 0.9]


class TestResumeAcrossGroupLayoutChange:
    """Same group count but different membership (activation params newly exempt)."""

    def test_two_group_state_remapped_by_orig_idx(self):
        from fluxflow_training.training.optimizer_factory import (
            remap_optimizer_state,
            split_weight_decay_groups,
        )

        m = _ActNet()
        cfg = {"type": "AdamW", "lr": 0.5, "weight_decay": 0.1}
        # HEAD layout: decay group = everything, no-decay = one scale scalar
        m.conv1_scale = nn.Parameter(torch.zeros(1))
        old = create_optimizer(split_weight_decay_groups(m.named_parameters()), cfg)
        assert len(old.param_groups) == 2
        for g in old.param_groups:
            for p in g["params"]:
                p.grad = torch.ones_like(p)
        old.step()
        saved = old.state_dict()
        new = create_optimizer(split_weight_decay_groups(m.named_parameters(), modules=[m]), cfg)
        assert len(new.param_groups) == 2
        remapped = remap_optimizer_state(new, saved)
        assert remapped is not None
        new.load_state_dict(remapped)
        for p in m.parameters():
            assert torch.equal(new.state[p]["exp_avg"], old.state[p]["exp_avg"])
        assert new.param_groups[1]["weight_decay"] == 0.0
