"""Unit tests for pipeline configuration and validation."""

import pytest

from fluxflow_training.training.pipeline_config import (
    OptimizerConfig,
    OptimizationConfig,
    PipelineConfig,
    PipelineConfigValidator,
    PipelineStepConfig,
    SchedulerConfig,
    TransitionCriteria,
    parse_pipeline_config,
)


class TestTransitionCriteria:
    """Test TransitionCriteria dataclass."""

    def test_default_epoch_mode(self):
        """Test default epoch-based transition."""
        criteria = TransitionCriteria(mode="epoch", value=10)
        assert criteria.mode == "epoch"
        assert criteria.value == 10
        assert criteria.metric is None
        assert criteria.threshold is None

    def test_loss_threshold_mode(self):
        """Test loss-threshold-based transition."""
        criteria = TransitionCriteria(
            mode="loss_threshold", metric="vae_loss", threshold=0.015, max_epochs=50
        )
        assert criteria.mode == "loss_threshold"
        assert criteria.metric == "vae_loss"
        assert criteria.threshold == 0.015
        assert criteria.max_epochs == 50


class TestOptimizerConfig:
    """Test OptimizerConfig dataclass."""

    def test_default_adamw(self):
        """Test default AdamW optimizer config."""
        config = OptimizerConfig()
        assert config.type == "AdamW"
        assert config.lr == 1e-5
        assert config.weight_decay == 0.0

    def test_custom_optimizer(self):
        """Test custom optimizer configuration."""
        config = OptimizerConfig(
            type="Lion", lr=5e-7, betas=(0.9, 0.95), weight_decay=0.01, decoupled_weight_decay=True
        )
        assert config.type == "Lion"
        assert config.lr == 5e-7
        assert config.betas == (0.9, 0.95)
        assert config.weight_decay == 0.01
        assert config.decoupled_weight_decay is True


class TestSchedulerConfig:
    """Test SchedulerConfig dataclass."""

    def test_default_cosine_annealing(self):
        """Test default CosineAnnealingLR config."""
        config = SchedulerConfig()
        assert config.type == "CosineAnnealingLR"
        assert config.eta_min_factor == 0.1

    def test_custom_scheduler(self):
        """Test custom scheduler configuration."""
        config = SchedulerConfig(type="StepLR", step_size=10000, gamma=0.5)
        assert config.type == "StepLR"
        assert config.step_size == 10000
        assert config.gamma == 0.5


class TestPipelineStepConfig:
    """Test PipelineStepConfig dataclass."""

    def test_minimal_step(self):
        """Test minimal valid step configuration."""
        step = PipelineStepConfig(name="test_step", n_epochs=10)
        assert step.name == "test_step"
        assert step.n_epochs == 10
        assert step.train_vae is False
        assert step.freeze == []

    def test_full_step_config(self):
        """Test fully configured step."""
        step = PipelineStepConfig(
            name="vae_training",
            n_epochs=50,
            description="Train VAE with SPADE",
            train_vae=True,
            gan_training=True,
            train_spade=True,
            freeze=["text_encoder"],
            batch_size=4,
            lr=2e-5,
            kl_beta=0.001,
        )
        assert step.name == "vae_training"
        assert step.n_epochs == 50
        assert step.train_vae is True
        assert step.gan_training is True
        assert step.train_spade is True
        assert step.freeze == ["text_encoder"]
        assert step.batch_size == 4
        assert step.lr == 2e-5
        assert step.kl_beta == 0.001


class TestParsePipelineConfig:
    """Test parse_pipeline_config function."""

    def test_parse_simple_pipeline(self):
        """Test parsing simple pipeline with one step."""
        config_dict = {
            "steps": [
                {
                    "name": "step1",
                    "n_epochs": 10,
                    "train_vae": True,
                }
            ]
        }

        pipeline = parse_pipeline_config(config_dict)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "step1"
        assert pipeline.steps[0].n_epochs == 10
        assert pipeline.steps[0].train_vae is True

    def test_parse_with_defaults(self):
        """Test parsing pipeline with defaults."""
        config_dict = {
            "defaults": {"batch_size": 8, "workers": 16},
            "steps": [
                {"name": "step1", "n_epochs": 10, "train_vae": True},
                {"name": "step2", "n_epochs": 5, "train_vae": True, "batch_size": 4},  # Override
            ],
        }

        pipeline = parse_pipeline_config(config_dict)
        assert pipeline.defaults is not None
        assert len(pipeline.steps) == 2

        # First step inherits defaults
        assert pipeline.steps[0].batch_size == 8
        assert pipeline.steps[0].workers == 16

        # Second step overrides batch_size
        assert pipeline.steps[1].batch_size == 4
        assert pipeline.steps[1].workers == 16

    def test_parse_with_inline_optimizers(self):
        """Test parsing with inline optimizer configs."""
        config_dict = {
            "steps": [
                {
                    "name": "step1",
                    "n_epochs": 10,
                    "train_vae": True,
                    "optimization": {
                        "optimizers": {
                            "vae": {"type": "AdamW", "lr": 1e-5, "betas": [0.9, 0.95]},
                            "discriminator": {
                                "type": "AdamW",
                                "lr": 1e-5,
                                "amsgrad": True,
                            },
                        },
                        "schedulers": {
                            "vae": {"type": "CosineAnnealingLR", "eta_min_factor": 0.1},
                        },
                    },
                }
            ]
        }

        pipeline = parse_pipeline_config(config_dict)
        step = pipeline.steps[0]

        assert step.optimization is not None
        assert "vae" in step.optimization.optimizers
        assert step.optimization.optimizers["vae"].type == "AdamW"
        assert step.optimization.optimizers["vae"].lr == 1e-5
        assert step.optimization.optimizers["vae"].betas == (0.9, 0.95)

        assert "discriminator" in step.optimization.optimizers
        assert step.optimization.optimizers["discriminator"].amsgrad is True

        assert "vae" in step.optimization.schedulers
        assert step.optimization.schedulers["vae"].type == "CosineAnnealingLR"

    def test_parse_transition_criteria(self):
        """Test parsing transition criteria."""
        config_dict = {
            "steps": [
                {
                    "name": "step1",
                    "n_epochs": 10,
                    "train_vae": True,
                    "transition_on": {"mode": "epoch", "value": 10},
                },
                {
                    "name": "step2",
                    "n_epochs": 20,
                    "train_vae": True,
                    "transition_on": {
                        "mode": "loss_threshold",
                        "metric": "vae_loss",
                        "threshold": 0.015,
                        "max_epochs": 30,
                    },
                },
            ]
        }

        pipeline = parse_pipeline_config(config_dict)

        # Step 1: epoch-based
        assert pipeline.steps[0].transition_on.mode == "epoch"
        assert pipeline.steps[0].transition_on.value == 10

        # Step 2: loss-threshold-based
        assert pipeline.steps[1].transition_on.mode == "loss_threshold"
        assert pipeline.steps[1].transition_on.metric == "vae_loss"
        assert pipeline.steps[1].transition_on.threshold == 0.015
        assert pipeline.steps[1].transition_on.max_epochs == 30


class TestPipelineConfigValidator:
    """Test PipelineConfigValidator."""

    def test_valid_single_step(self):
        """Test validation of valid single-step pipeline."""
        pipeline = PipelineConfig(
            steps=[
                PipelineStepConfig(
                    name="vae_training",
                    n_epochs=10,
                    train_vae=True,
                    freeze=["text_encoder"],
                    transition_on=TransitionCriteria(mode="epoch", value=10),
                )
            ]
        )

        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert len(errors) == 0

    def test_valid_multi_step(self):
        """Test validation of valid multi-step pipeline."""
        pipeline = PipelineConfig(
            steps=[
                PipelineStepConfig(
                    name="step1",
                    n_epochs=5,
                    train_vae=True,
                    transition_on=TransitionCriteria(mode="epoch", value=5),
                ),
                PipelineStepConfig(
                    name="step2",
                    n_epochs=10,
                    train_vae=True,
                    gan_training=True,
                    transition_on=TransitionCriteria(
                        mode="loss_threshold",
                        metric="vae_loss",
                        threshold=0.015,
                        max_epochs=15,
                    ),
                ),
            ]
        )

        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert len(errors) == 0

    def test_empty_pipeline(self):
        """Test validation fails for empty pipeline."""
        pipeline = PipelineConfig(steps=[])
        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert len(errors) == 1
        assert "no steps defined" in errors[0].lower()

    def test_zero_epochs(self):
        """Test validation fails for zero epochs."""
        pipeline = PipelineConfig(
            steps=[PipelineStepConfig(name="test", n_epochs=0, train_vae=True)]
        )
        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert any("n_epochs must be > 0" in e for e in errors)

    def test_no_training_modes(self):
        """Test validation fails when no training modes enabled."""
        pipeline = PipelineConfig(steps=[PipelineStepConfig(name="test", n_epochs=10)])
        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert any("at least one training mode must be enabled" in e.lower() for e in errors)

    def test_freeze_unfreeze_conflict(self):
        """Test validation detects freeze/train conflicts."""
        pipeline = PipelineConfig(
            steps=[
                PipelineStepConfig(
                    name="conflict_step",
                    n_epochs=10,
                    train_vae=True,
                    freeze=["compressor", "expander"],  # Conflict: training VAE but freezing it
                )
            ]
        )

        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert any("cannot freeze and train the same component" in e.lower() for e in errors)
        assert any("compressor" in e for e in errors)
        assert any("expander" in e for e in errors)

    def test_invalid_component_name(self):
        """Test validation detects invalid component names."""
        pipeline = PipelineConfig(
            steps=[
                PipelineStepConfig(
                    name="invalid",
                    n_epochs=10,
                    train_vae=True,
                    freeze=["invalid_component", "another_bad_name"],
                )
            ]
        )

        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert any("unknown component" in e.lower() for e in errors)

    def test_invalid_optimizer_type(self):
        """Test validation detects invalid optimizer types."""
        opt_config = OptimizationConfig(
            optimizers={"vae": OptimizerConfig(type="InvalidOptimizer")}
        )

        pipeline = PipelineConfig(
            steps=[
                PipelineStepConfig(
                    name="test", n_epochs=10, train_vae=True, optimization=opt_config
                )
            ]
        )

        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert any("unknown optimizer type" in e.lower() for e in errors)

    def test_invalid_scheduler_type(self):
        """Test validation detects invalid scheduler types."""
        opt_config = OptimizationConfig(
            schedulers={"vae": SchedulerConfig(type="InvalidScheduler")}
        )

        pipeline = PipelineConfig(
            steps=[
                PipelineStepConfig(
                    name="test", n_epochs=10, train_vae=True, optimization=opt_config
                )
            ]
        )

        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert any("unknown scheduler type" in e.lower() for e in errors)

    def test_epoch_transition_missing_value(self):
        """Test validation fails for epoch transition without value."""
        pipeline = PipelineConfig(
            steps=[
                PipelineStepConfig(
                    name="test",
                    n_epochs=10,
                    train_vae=True,
                    transition_on=TransitionCriteria(mode="epoch"),  # Missing value
                )
            ]
        )

        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert any("epoch-based transition requires 'value'" in e.lower() for e in errors)

    def test_loss_threshold_missing_metric(self):
        """Test validation fails for loss-threshold without metric."""
        pipeline = PipelineConfig(
            steps=[
                PipelineStepConfig(
                    name="test",
                    n_epochs=10,
                    train_vae=True,
                    transition_on=TransitionCriteria(
                        mode="loss_threshold",
                        threshold=0.015,
                        max_epochs=20,
                        # Missing metric
                    ),
                )
            ]
        )

        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert any("loss-threshold transition requires 'metric'" in e.lower() for e in errors)

    def test_loss_threshold_missing_threshold(self):
        """Test validation fails for loss-threshold without threshold."""
        pipeline = PipelineConfig(
            steps=[
                PipelineStepConfig(
                    name="test",
                    n_epochs=10,
                    train_vae=True,
                    transition_on=TransitionCriteria(
                        mode="loss_threshold",
                        metric="vae_loss",
                        max_epochs=20,
                        # Missing threshold
                    ),
                )
            ]
        )

        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert any("loss-threshold transition requires 'threshold'" in e.lower() for e in errors)

    def test_loss_threshold_missing_max_epochs(self):
        """Test validation fails for loss-threshold without max_epochs."""
        pipeline = PipelineConfig(
            steps=[
                PipelineStepConfig(
                    name="test",
                    n_epochs=10,
                    train_vae=True,
                    transition_on=TransitionCriteria(
                        mode="loss_threshold",
                        metric="vae_loss",
                        threshold=0.015,
                        # Missing max_epochs
                    ),
                )
            ]
        )

        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert any("loss-threshold transition requires 'max_epochs'" in e.lower() for e in errors)

    def test_loss_threshold_invalid_metric(self):
        """Test validation detects invalid metric names."""
        pipeline = PipelineConfig(
            steps=[
                PipelineStepConfig(
                    name="test",
                    n_epochs=10,
                    train_vae=True,
                    transition_on=TransitionCriteria(
                        mode="loss_threshold",
                        metric="invalid_metric_name",
                        threshold=0.015,
                        max_epochs=20,
                    ),
                )
            ]
        )

        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert any("unknown metric" in e.lower() for e in errors)

    def test_multiple_errors_reported(self):
        """Test validator reports all errors, not just the first."""
        pipeline = PipelineConfig(
            steps=[
                PipelineStepConfig(
                    name="bad_step",
                    n_epochs=0,  # Error 1
                    # No training modes  # Error 2
                    freeze=["invalid_component"],  # Error 3
                )
            ]
        )

        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert len(errors) >= 3  # Should report multiple errors

    def test_valid_freeze_without_conflict(self):
        """Test freezing text_encoder while training VAE is valid."""
        pipeline = PipelineConfig(
            steps=[
                PipelineStepConfig(
                    name="vae_train",
                    n_epochs=10,
                    train_vae=True,
                    freeze=["text_encoder"],  # Valid: text_encoder not part of VAE
                    transition_on=TransitionCriteria(mode="epoch", value=10),
                )
            ]
        )

        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert len(errors) == 0


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_empty_freeze_list(self):
        """Test empty freeze list is valid."""
        pipeline = PipelineConfig(
            steps=[
                PipelineStepConfig(
                    name="test",
                    n_epochs=10,
                    train_vae=True,
                    freeze=[],  # Empty list
                    transition_on=TransitionCriteria(mode="epoch", value=10),
                )
            ]
        )

        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert len(errors) == 0

    def test_multiple_training_modes(self):
        """Test multiple training modes enabled simultaneously."""
        pipeline = PipelineConfig(
            steps=[
                PipelineStepConfig(
                    name="full_training",
                    n_epochs=10,
                    train_vae=True,
                    gan_training=True,
                    train_spade=True,
                    train_diff_full=True,
                    transition_on=TransitionCriteria(mode="epoch", value=10),
                )
            ]
        )

        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert len(errors) == 0

    def test_gan_with_discriminator_frozen(self):
        """Test GAN training with discriminator frozen (should conflict)."""
        pipeline = PipelineConfig(
            steps=[
                PipelineStepConfig(
                    name="conflict",
                    n_epochs=10,
                    gan_training=True,
                    freeze=["discriminator"],  # Conflict
                )
            ]
        )

        validator = PipelineConfigValidator(pipeline)
        errors = validator.validate()
        assert any("cannot freeze and train the same component" in e.lower() for e in errors)
        assert any("discriminator" in e for e in errors)
