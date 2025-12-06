"""Unit tests for scheduler factory (fluxflow_training/training/scheduler_factory.py)."""

import pytest
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    ExponentialLR,
    LinearLR,
    ReduceLROnPlateau,
    StepLR,
)

from fluxflow_training.training.scheduler_factory import (
    SUPPORTED_SCHEDULERS,
    create_scheduler,
    get_default_scheduler_config,
    validate_scheduler_config,
)


class TestValidateSchedulerConfig:
    """Tests for validate_scheduler_config function."""

    def test_valid_cosine_annealing_config(self):
        """Valid CosineAnnealingLR config should pass validation."""
        config = {"type": "CosineAnnealingLR", "eta_min_factor": 0.1}
        validate_scheduler_config(config, total_steps=1000)  # Should not raise

    def test_valid_linear_config(self):
        """Valid LinearLR config should pass validation."""
        config = {
            "type": "LinearLR",
            "start_factor": 1.0,
            "end_factor": 0.1,
        }
        validate_scheduler_config(config, total_steps=1000)  # Should not raise

    def test_valid_exponential_config(self):
        """Valid ExponentialLR config should pass validation."""
        config = {"type": "ExponentialLR", "gamma": 0.95}
        validate_scheduler_config(config, total_steps=1000)  # Should not raise

    def test_valid_step_config(self):
        """Valid StepLR config should pass validation."""
        config = {"type": "StepLR", "step_size": 100, "gamma": 0.5}
        validate_scheduler_config(config, total_steps=1000)  # Should not raise

    def test_valid_constant_config(self):
        """Valid ConstantLR config should pass validation."""
        config = {"type": "ConstantLR", "factor": 0.5}
        validate_scheduler_config(config, total_steps=1000)  # Should not raise

    def test_valid_reduce_on_plateau_config(self):
        """Valid ReduceLROnPlateau config should pass validation."""
        config = {
            "type": "ReduceLROnPlateau",
            "factor": 0.5,
            "patience": 10,
        }
        validate_scheduler_config(config, total_steps=1000)  # Should not raise

    def test_zero_total_steps_raises(self):
        """Zero total steps should raise ValueError."""
        config = {"type": "CosineAnnealingLR"}
        with pytest.raises(ValueError, match="Total steps must be positive"):
            validate_scheduler_config(config, total_steps=0)

    def test_negative_total_steps_raises(self):
        """Negative total steps should raise ValueError."""
        config = {"type": "CosineAnnealingLR"}
        with pytest.raises(ValueError, match="Total steps must be positive"):
            validate_scheduler_config(config, total_steps=-100)

    def test_zero_eta_min_factor_raises(self):
        """Zero eta_min_factor should raise ValueError."""
        config = {"type": "CosineAnnealingLR", "eta_min_factor": 0}
        with pytest.raises(ValueError, match="eta_min_factor must be in"):
            validate_scheduler_config(config, total_steps=1000)

    def test_eta_min_factor_greater_than_one_raises(self):
        """eta_min_factor greater than 1 should raise ValueError."""
        config = {"type": "CosineAnnealingLR", "eta_min_factor": 2.0}
        with pytest.raises(ValueError, match="eta_min_factor must be in"):
            validate_scheduler_config(config, total_steps=1000)

    def test_negative_eta_min_factor_raises(self):
        """Negative eta_min_factor should raise ValueError."""
        config = {"type": "CosineAnnealingLR", "eta_min_factor": -0.1}
        with pytest.raises(ValueError, match="eta_min_factor must be in"):
            validate_scheduler_config(config, total_steps=1000)

    def test_gamma_equals_one_raises(self):
        """Gamma equal to 1.0 should raise ValueError."""
        config = {"type": "ExponentialLR", "gamma": 1.0}
        with pytest.raises(ValueError, match="Gamma must be in"):
            validate_scheduler_config(config, total_steps=1000)

    def test_gamma_greater_than_one_raises(self):
        """Gamma greater than 1.0 should raise ValueError."""
        config = {"type": "StepLR", "gamma": 1.5}
        with pytest.raises(ValueError, match="Gamma must be in"):
            validate_scheduler_config(config, total_steps=1000)

    def test_zero_gamma_raises(self):
        """Zero gamma should raise ValueError."""
        config = {"type": "ExponentialLR", "gamma": 0}
        with pytest.raises(ValueError, match="Gamma must be in"):
            validate_scheduler_config(config, total_steps=1000)

    def test_zero_step_size_raises(self):
        """Zero step_size should raise ValueError."""
        config = {"type": "StepLR", "step_size": 0, "gamma": 0.5}
        with pytest.raises(ValueError, match="Step size must be positive"):
            validate_scheduler_config(config, total_steps=1000)

    def test_negative_step_size_raises(self):
        """Negative step_size should raise ValueError."""
        config = {"type": "StepLR", "step_size": -10, "gamma": 0.5}
        with pytest.raises(ValueError, match="Step size must be positive"):
            validate_scheduler_config(config, total_steps=1000)

    def test_zero_start_factor_raises(self):
        """Zero start_factor should raise ValueError."""
        config = {"type": "LinearLR", "start_factor": 0}
        with pytest.raises(ValueError, match="Start factor must be positive"):
            validate_scheduler_config(config, total_steps=1000)

    def test_zero_end_factor_raises(self):
        """Zero end_factor should raise ValueError."""
        config = {"type": "LinearLR", "end_factor": 0}
        with pytest.raises(ValueError, match="End factor must be positive"):
            validate_scheduler_config(config, total_steps=1000)

    def test_reduce_plateau_factor_equals_one_raises(self):
        """ReduceLROnPlateau factor equal to 1.0 should raise ValueError."""
        config = {"type": "ReduceLROnPlateau", "factor": 1.0}
        with pytest.raises(ValueError, match="ReduceLROnPlateau factor must be < 1"):
            validate_scheduler_config(config, total_steps=1000)

    def test_reduce_plateau_factor_greater_than_one_raises(self):
        """ReduceLROnPlateau factor greater than 1.0 should raise ValueError."""
        config = {"type": "ReduceLROnPlateau", "factor": 1.5}
        with pytest.raises(ValueError, match="ReduceLROnPlateau factor must be < 1"):
            validate_scheduler_config(config, total_steps=1000)

    def test_negative_patience_raises(self):
        """Negative patience should raise ValueError."""
        config = {"type": "ReduceLROnPlateau", "factor": 0.5, "patience": -1}
        with pytest.raises(ValueError, match="Patience must be non-negative"):
            validate_scheduler_config(config, total_steps=1000)


class TestCreateScheduler:
    """Tests for create_scheduler function."""

    @pytest.fixture
    def optimizer(self):
        """Create a simple optimizer for testing."""
        model = nn.Linear(10, 5)
        return optim.AdamW(model.parameters(), lr=1e-4)

    def test_create_cosine_annealing(self, optimizer):
        """Should create CosineAnnealingLR scheduler."""
        config = {"type": "CosineAnnealingLR", "eta_min_factor": 0.1}
        scheduler = create_scheduler(optimizer, config, total_steps=1000)
        assert isinstance(scheduler, CosineAnnealingLR)
        assert scheduler.T_max == 1000

    def test_create_linear(self, optimizer):
        """Should create LinearLR scheduler."""
        config = {"type": "LinearLR", "start_factor": 1.0, "end_factor": 0.1}
        scheduler = create_scheduler(optimizer, config, total_steps=1000)
        assert isinstance(scheduler, LinearLR)

    def test_create_exponential(self, optimizer):
        """Should create ExponentialLR scheduler."""
        config = {"type": "ExponentialLR", "gamma": 0.95}
        scheduler = create_scheduler(optimizer, config, total_steps=1000)
        assert isinstance(scheduler, ExponentialLR)

    def test_create_step(self, optimizer):
        """Should create StepLR scheduler."""
        config = {"type": "StepLR", "step_size": 100, "gamma": 0.5}
        scheduler = create_scheduler(optimizer, config, total_steps=1000)
        assert isinstance(scheduler, StepLR)
        assert scheduler.step_size == 100
        assert scheduler.gamma == 0.5

    def test_create_constant(self, optimizer):
        """Should create ConstantLR scheduler."""
        config = {"type": "ConstantLR", "factor": 0.5}
        scheduler = create_scheduler(optimizer, config, total_steps=1000)
        assert isinstance(scheduler, ConstantLR)

    def test_create_reduce_on_plateau(self, optimizer):
        """Should create ReduceLROnPlateau scheduler."""
        config = {"type": "ReduceLROnPlateau", "factor": 0.5, "patience": 5}
        scheduler = create_scheduler(optimizer, config, total_steps=1000)
        assert isinstance(scheduler, ReduceLROnPlateau)
        assert scheduler.factor == 0.5
        assert scheduler.patience == 5

    def test_unsupported_scheduler_raises(self, optimizer):
        """Unsupported scheduler type should raise ValueError."""
        config = {"type": "InvalidScheduler"}
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            create_scheduler(optimizer, config, total_steps=1000)

    def test_eta_min_calculated_correctly(self, optimizer):
        """eta_min should be calculated from initial LR and eta_min_factor."""
        config = {"type": "CosineAnnealingLR", "eta_min_factor": 0.1}
        scheduler = create_scheduler(optimizer, config, total_steps=1000)
        # Initial LR is 1e-4, so eta_min should be 1e-5
        assert scheduler.eta_min == pytest.approx(1e-5)

    def test_default_type_used(self, optimizer):
        """Default scheduler type should be used if not specified."""
        config = {}
        scheduler = create_scheduler(optimizer, config, total_steps=1000)
        assert isinstance(scheduler, CosineAnnealingLR)


class TestGetDefaultSchedulerConfig:
    """Tests for get_default_scheduler_config function."""

    def test_flow_default(self):
        """Flow model should use CosineAnnealingLR."""
        config = get_default_scheduler_config("flow")
        assert config["type"] == "CosineAnnealingLR"
        assert config["eta_min_factor"] == 0.1

    def test_vae_default(self):
        """VAE model should use CosineAnnealingLR."""
        config = get_default_scheduler_config("vae")
        assert config["type"] == "CosineAnnealingLR"
        assert config["eta_min_factor"] == 0.1

    def test_text_encoder_default(self):
        """Text encoder should use more aggressive decay."""
        config = get_default_scheduler_config("text_encoder")
        assert config["type"] == "CosineAnnealingLR"
        assert config["eta_min_factor"] == 0.001  # More aggressive

    def test_discriminator_default(self):
        """Discriminator should use CosineAnnealingLR."""
        config = get_default_scheduler_config("discriminator")
        assert config["type"] == "CosineAnnealingLR"
        assert config["eta_min_factor"] == 0.1

    def test_unknown_model_uses_vae_default(self):
        """Unknown model name should use VAE defaults."""
        config = get_default_scheduler_config("unknown_model")
        assert config["type"] == "CosineAnnealingLR"
        assert config["eta_min_factor"] == 0.1

    def test_all_defaults_pass_validation(self):
        """All default configs should pass validation."""
        for model_name in ["flow", "vae", "text_encoder", "discriminator"]:
            config = get_default_scheduler_config(model_name)
            validate_scheduler_config(config, total_steps=1000)  # Should not raise


class TestSupportedSchedulers:
    """Tests for SUPPORTED_SCHEDULERS constant."""

    def test_contains_expected_schedulers(self):
        """Should contain all expected scheduler types."""
        expected = [
            "CosineAnnealingLR",
            "LinearLR",
            "ExponentialLR",
            "ConstantLR",
            "StepLR",
            "ReduceLROnPlateau",
        ]
        for sched_name in expected:
            assert sched_name in SUPPORTED_SCHEDULERS

    def test_scheduler_classes_are_correct(self):
        """Scheduler classes should be correct PyTorch types."""
        assert SUPPORTED_SCHEDULERS["CosineAnnealingLR"] == CosineAnnealingLR
        assert SUPPORTED_SCHEDULERS["LinearLR"] == LinearLR
        assert SUPPORTED_SCHEDULERS["ExponentialLR"] == ExponentialLR
        assert SUPPORTED_SCHEDULERS["ConstantLR"] == ConstantLR
        assert SUPPORTED_SCHEDULERS["StepLR"] == StepLR
        assert SUPPORTED_SCHEDULERS["ReduceLROnPlateau"] == ReduceLROnPlateau


class TestSchedulerBehavior:
    """Integration tests for scheduler behavior."""

    @pytest.fixture
    def optimizer(self):
        """Create a simple optimizer for testing."""
        model = nn.Linear(10, 5)
        return optim.AdamW(model.parameters(), lr=1e-3)

    def test_cosine_annealing_decay(self, optimizer):
        """CosineAnnealingLR should decay LR over time."""
        config = {"type": "CosineAnnealingLR", "eta_min_factor": 0.1}
        scheduler = create_scheduler(optimizer, config, total_steps=100)

        initial_lr = optimizer.param_groups[0]["lr"]

        # Step through half the schedule
        for _ in range(50):
            scheduler.step()

        mid_lr = optimizer.param_groups[0]["lr"]
        assert mid_lr < initial_lr

        # Step to the end
        for _ in range(50):
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]
        # Final LR should be close to eta_min
        assert final_lr == pytest.approx(initial_lr * 0.1, rel=0.01)

    def test_step_lr_decay(self, optimizer):
        """StepLR should decay LR at step intervals."""
        config = {"type": "StepLR", "step_size": 10, "gamma": 0.5}
        scheduler = create_scheduler(optimizer, config, total_steps=100)

        initial_lr = optimizer.param_groups[0]["lr"]

        # Step 10 times (one decay)
        for _ in range(10):
            scheduler.step()

        after_one_decay = optimizer.param_groups[0]["lr"]
        assert after_one_decay == pytest.approx(initial_lr * 0.5)

        # Step 10 more times (second decay)
        for _ in range(10):
            scheduler.step()

        after_two_decays = optimizer.param_groups[0]["lr"]
        assert after_two_decays == pytest.approx(initial_lr * 0.25)
