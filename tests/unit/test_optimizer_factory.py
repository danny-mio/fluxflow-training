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
        """Flow model should use Lion optimizer."""
        config = get_default_optimizer_config("flow")
        assert config["type"] == "Lion"
        assert config["lr"] == 5e-7

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
