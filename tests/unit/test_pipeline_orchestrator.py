"""Unit tests for TrainingPipelineOrchestrator."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from fluxflow_training.training.checkpoint_manager import CheckpointManager
from fluxflow_training.training.pipeline_config import (
    PipelineConfig,
    PipelineStepConfig,
    TransitionCriteria,
    parse_pipeline_config,
)
from fluxflow_training.training.pipeline_orchestrator import TrainingPipelineOrchestrator


@pytest.fixture
def mock_models():
    """Create mock model dictionary."""
    models = {}
    for name in ["compressor", "expander", "flow_processor", "text_encoder", "discriminator"]:
        model = Mock(spec=nn.Module)
        model.parameters = Mock(return_value=[Mock(spec=nn.Parameter, numel=Mock(return_value=1000))])
        models[name] = model
    return models


@pytest.fixture
def mock_checkpoint_manager(tmp_path):
    """Create mock checkpoint manager."""
    return Mock(spec=CheckpointManager)


@pytest.fixture
def simple_pipeline_config():
    """Create simple pipeline config for testing."""
    config_dict = {
        "steps": [
            {
                "name": "step1",
                "n_epochs": 10,
                "train_vae": True,
                "freeze": ["text_encoder"],
                "transition_on": {"mode": "epoch", "value": 10},
            },
            {
                "name": "step2",
                "n_epochs": 5,
                "train_vae": True,
                "gan_training": True,
                "transition_on": {
                    "mode": "loss_threshold",
                    "metric": "vae_loss",
                    "threshold": 0.015,
                    "max_epochs": 10,
                },
            },
        ]
    }
    return parse_pipeline_config(config_dict)


class TestOrchestratorInitialization:
    """Test orchestrator initialization."""

    def test_initialization_success(
        self, simple_pipeline_config, mock_models, mock_checkpoint_manager
    ):
        """Test successful initialization."""
        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        assert orchestrator.current_step_index == 0
        assert orchestrator.global_step == 0
        assert orchestrator.steps_completed == []
        assert orchestrator.step_metrics == {}

    def test_initialization_missing_model(self, simple_pipeline_config, mock_checkpoint_manager):
        """Test initialization fails with missing model."""
        incomplete_models = {"compressor": Mock(), "expander": Mock()}

        with pytest.raises(ValueError, match="Missing required model components"):
            TrainingPipelineOrchestrator(
                config=simple_pipeline_config,
                models=incomplete_models,
                checkpoint_manager=mock_checkpoint_manager,
                accelerator=Mock(),
                dataloader=Mock(),
                dataset=Mock(),
            )


class TestModelFreezing:
    """Test model freeze/unfreeze functionality."""

    def test_freeze_model(self, simple_pipeline_config, mock_models, mock_checkpoint_manager):
        """Test freezing a model."""
        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        # Create real model for testing
        model = nn.Linear(10, 10)
        orchestrator.models["test_model"] = model

        # Initially, parameters are trainable
        assert all(p.requires_grad for p in model.parameters())

        # Freeze the model
        orchestrator.freeze_model("test_model")

        # After freezing, parameters should not require grad
        assert all(not p.requires_grad for p in model.parameters())

    def test_unfreeze_model(self, simple_pipeline_config, mock_models, mock_checkpoint_manager):
        """Test unfreezing a model."""
        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        # Create real model for testing
        model = nn.Linear(10, 10)
        orchestrator.models["test_model"] = model

        # Freeze then unfreeze
        for param in model.parameters():
            param.requires_grad = False

        orchestrator.unfreeze_model("test_model")

        # After unfreezing, parameters should require grad
        assert all(p.requires_grad for p in model.parameters())

    def test_freeze_nonexistent_model(
        self, simple_pipeline_config, mock_models, mock_checkpoint_manager
    ):
        """Test freezing nonexistent model logs warning and doesn't crash."""
        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        # Should not raise, just log warning
        orchestrator.freeze_model("nonexistent_model")


class TestConfigureStepModels:
    """Test model configuration for pipeline steps."""

    def test_configure_freeze_list(
        self, simple_pipeline_config, mock_models, mock_checkpoint_manager
    ):
        """Test configuring models with freeze list."""
        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        step = PipelineStepConfig(
            name="test",
            n_epochs=10,
            train_vae=True,
            freeze=["text_encoder", "discriminator"],
            transition_on=TransitionCriteria(mode="epoch", value=10),
        )

        # Add real models for testing
        for name in ["text_encoder", "discriminator", "compressor"]:
            orchestrator.models[name] = nn.Linear(10, 10)

        orchestrator.configure_step_models(step)

        # text_encoder and discriminator should be frozen
        assert all(not p.requires_grad for p in orchestrator.models["text_encoder"].parameters())
        assert all(not p.requires_grad for p in orchestrator.models["discriminator"].parameters())

        # compressor should still be trainable
        assert all(p.requires_grad for p in orchestrator.models["compressor"].parameters())


class TestMetricTracking:
    """Test metric tracking for loss-threshold transitions."""

    def test_update_metrics(self, simple_pipeline_config, mock_models, mock_checkpoint_manager):
        """Test updating metrics."""
        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        losses = {"vae_loss": 0.5, "kl_loss": 2.0}
        orchestrator.update_metrics("step1", losses)

        assert "step1" in orchestrator.step_metrics
        assert "vae_loss" in orchestrator.step_metrics["step1"]
        assert "kl_loss" in orchestrator.step_metrics["step1"]
        assert orchestrator.step_metrics["step1"]["vae_loss"] == [0.5]
        assert orchestrator.step_metrics["step1"]["kl_loss"] == [2.0]

    def test_update_metrics_multiple_times(
        self, simple_pipeline_config, mock_models, mock_checkpoint_manager
    ):
        """Test multiple metric updates."""
        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        for i in range(5):
            orchestrator.update_metrics("step1", {"loss": float(i)})

        assert len(orchestrator.step_metrics["step1"]["loss"]) == 5
        assert orchestrator.step_metrics["step1"]["loss"] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_metric_buffer_limit(
        self, simple_pipeline_config, mock_models, mock_checkpoint_manager
    ):
        """Test metric buffer maintains max 100 items."""
        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        # Add 150 metrics
        for i in range(150):
            orchestrator.update_metrics("step1", {"loss": float(i)})

        # Should keep only last 100
        assert len(orchestrator.step_metrics["step1"]["loss"]) == 100
        assert orchestrator.step_metrics["step1"]["loss"][0] == 50.0  # First 50 removed
        assert orchestrator.step_metrics["step1"]["loss"][-1] == 149.0


class TestSmoothedMetrics:
    """Test smoothed metric calculation."""

    def test_get_smoothed_metric(
        self, simple_pipeline_config, mock_models, mock_checkpoint_manager
    ):
        """Test getting smoothed metric value."""
        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        # Add 30 metrics with values 0-29
        for i in range(30):
            orchestrator.update_metrics("step1", {"loss": float(i)})

        # Get smoothed value over last 20 (window=20)
        smoothed = orchestrator.get_smoothed_metric("step1", "loss", window=20)

        # Should be average of 10-29 (last 20 values)
        expected = sum(range(10, 30)) / 20.0
        assert smoothed == pytest.approx(expected)

    def test_get_smoothed_metric_insufficient_data(
        self, simple_pipeline_config, mock_models, mock_checkpoint_manager
    ):
        """Test smoothed metric returns None with insufficient data."""
        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        # Add only 10 metrics
        for i in range(10):
            orchestrator.update_metrics("step1", {"loss": float(i)})

        # Request window=20, should return None
        smoothed = orchestrator.get_smoothed_metric("step1", "loss", window=20)
        assert smoothed is None

    def test_get_smoothed_metric_nonexistent(
        self, simple_pipeline_config, mock_models, mock_checkpoint_manager
    ):
        """Test smoothed metric for nonexistent step/metric."""
        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        smoothed = orchestrator.get_smoothed_metric("nonexistent_step", "loss")
        assert smoothed is None


class TestTransitionCriteria:
    """Test transition criteria evaluation."""

    def test_should_transition_epoch_mode(
        self, simple_pipeline_config, mock_models, mock_checkpoint_manager
    ):
        """Test epoch-based transition."""
        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        step = PipelineStepConfig(
            name="test",
            n_epochs=10,
            train_vae=True,
            transition_on=TransitionCriteria(mode="epoch", value=10),
        )

        # Before reaching target
        should_trans, reason = orchestrator.should_transition(step, current_epoch=5)
        assert should_trans is False
        assert "5/10" in reason

        # After reaching target
        should_trans, reason = orchestrator.should_transition(step, current_epoch=10)
        assert should_trans is True
        assert "Completed 10 epochs" in reason

    def test_should_transition_loss_threshold_mode(
        self, simple_pipeline_config, mock_models, mock_checkpoint_manager
    ):
        """Test loss-threshold-based transition."""
        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        step = PipelineStepConfig(
            name="test",
            n_epochs=50,
            train_vae=True,
            transition_on=TransitionCriteria(
                mode="loss_threshold", metric="vae_loss", threshold=0.01, max_epochs=50
            ),
        )

        # Add metrics above threshold
        for i in range(30):
            orchestrator.update_metrics("test", {"vae_loss": 0.02})  # Above 0.01

        should_trans, reason = orchestrator.should_transition(step, current_epoch=10)
        assert should_trans is False
        assert "0.0200" in reason  # Shows current value

        # Add metrics below threshold
        for i in range(30):
            orchestrator.update_metrics("test", {"vae_loss": 0.005})  # Below 0.01

        should_trans, reason = orchestrator.should_transition(step, current_epoch=15)
        assert should_trans is True
        assert "threshold met" in reason.lower()

    def test_should_transition_max_epochs_reached(
        self, simple_pipeline_config, mock_models, mock_checkpoint_manager
    ):
        """Test loss-threshold transitions at max_epochs limit."""
        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        step = PipelineStepConfig(
            name="test",
            n_epochs=50,
            train_vae=True,
            transition_on=TransitionCriteria(
                mode="loss_threshold", metric="vae_loss", threshold=0.001, max_epochs=20
            ),
        )

        # Add metrics but threshold not met
        for i in range(30):
            orchestrator.update_metrics("test", {"vae_loss": 0.05})  # Well above threshold

        # Should transition due to max_epochs
        should_trans, reason = orchestrator.should_transition(step, current_epoch=20)
        assert should_trans is True
        assert "Max epochs (20) reached" in reason


class TestPipelineMetadata:
    """Test pipeline metadata generation."""

    def test_get_pipeline_metadata(
        self, simple_pipeline_config, mock_models, mock_checkpoint_manager
    ):
        """Test generating pipeline metadata."""
        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        orchestrator.steps_completed = ["step1"]
        metadata = orchestrator.get_pipeline_metadata(step_index=1, epoch=15)

        assert metadata["current_step_index"] == 1
        assert metadata["current_step_name"] == "step2"
        assert metadata["total_steps"] == 2
        assert metadata["steps_completed"] == ["step1"]
        assert metadata["step_start_epoch"] == 15


class TestResumeFromCheckpoint:
    """Test resuming from checkpoint."""

    def test_resume_no_checkpoint(
        self, simple_pipeline_config, mock_models, mock_checkpoint_manager
    ):
        """Test resume with no checkpoint."""
        mock_checkpoint_manager.load_training_state.return_value = None

        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        step_idx, epoch, batch_idx = orchestrator.resume_from_checkpoint()

        assert step_idx == 0
        assert epoch == 0
        assert batch_idx == 0

    def test_resume_legacy_checkpoint(
        self, simple_pipeline_config, mock_models, mock_checkpoint_manager
    ):
        """Test resume from legacy (non-pipeline) checkpoint."""
        mock_checkpoint_manager.load_training_state.return_value = {
            "mode": "legacy",
            "epoch": 10,
            "batch_idx": 50,
            "global_step": 1000,
        }

        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        step_idx, epoch, batch_idx = orchestrator.resume_from_checkpoint()

        # Should start from beginning for legacy checkpoints
        assert step_idx == 0
        assert epoch == 0
        assert batch_idx == 0

    def test_resume_pipeline_checkpoint(
        self, simple_pipeline_config, mock_models, mock_checkpoint_manager
    ):
        """Test resume from pipeline checkpoint."""
        mock_checkpoint_manager.load_training_state.return_value = {
            "mode": "pipeline",
            "epoch": 15,
            "batch_idx": 75,
            "global_step": 2500,
            "pipeline": {
                "current_step_index": 1,
                "current_step_name": "step2",
                "total_steps": 2,
                "steps_completed": ["step1"],
            },
        }

        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        step_idx, epoch, batch_idx = orchestrator.resume_from_checkpoint()

        assert step_idx == 1
        assert epoch == 15
        assert batch_idx == 75
        assert orchestrator.global_step == 2500
        assert orchestrator.steps_completed == ["step1"]


class TestPrintPipelineSummary:
    """Test pipeline summary printing."""

    def test_print_pipeline_summary(
        self, simple_pipeline_config, mock_models, mock_checkpoint_manager, capsys
    ):
        """Test printing pipeline summary."""
        orchestrator = TrainingPipelineOrchestrator(
            config=simple_pipeline_config,
            models=mock_models,
            checkpoint_manager=mock_checkpoint_manager,
            accelerator=Mock(),
            dataloader=Mock(),
            dataset=Mock(),
        )

        orchestrator.print_pipeline_summary()

        captured = capsys.readouterr()
        assert "PIPELINE EXECUTION PLAN" in captured.out
        assert "Step 1/2: step1" in captured.out
        assert "Step 2/2: step2" in captured.out
        assert "Total epochs: 15" in captured.out  # 10 + 5
