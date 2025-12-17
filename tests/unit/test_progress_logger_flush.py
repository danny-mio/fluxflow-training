"""Unit tests for TrainingProgressLogger flush policies and error handling."""

import json
import tempfile
from pathlib import Path

import pytest

from fluxflow_training.training.progress_logger import TrainingProgressLogger


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_flush_policy_immediate(temp_output_dir):
    """Test immediate flush policy writes immediately."""
    logger = TrainingProgressLogger(
        str(temp_output_dir),
        flush_policy="immediate",
    )

    logger.log_metrics(epoch=0, batch=1, global_step=1, metrics={"loss": 1.0})

    # File should exist and have content
    metrics_file = temp_output_dir / "graph" / "training_metrics.jsonl"
    assert metrics_file.exists()
    with open(metrics_file) as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["metrics"]["loss"] == 1.0


def test_flush_policy_periodic(temp_output_dir):
    """Test periodic flush policy flushes after N writes."""
    logger = TrainingProgressLogger(
        str(temp_output_dir),
        flush_policy="periodic",
        flush_interval=3,
    )

    logger.log_metrics(epoch=0, batch=1, global_step=1, metrics={"loss": 1.0})
    logger.log_metrics(epoch=0, batch=2, global_step=2, metrics={"loss": 0.9})
    assert logger._write_count == 2

    logger.log_metrics(epoch=0, batch=3, global_step=3, metrics={"loss": 0.8})
    assert logger._write_count == 0  # Should reset after flush


def test_flush_policy_buffered(temp_output_dir):
    """Test buffered flush policy doesn't flush explicitly."""
    logger = TrainingProgressLogger(
        str(temp_output_dir),
        flush_policy="buffered",
    )

    logger.log_metrics(epoch=0, batch=1, global_step=1, metrics={"loss": 1.0})
    # Buffered mode doesn't increment write count or flush
    # Just verify it doesn't error
    assert logger.flush_policy == "buffered"


def test_sanitize_nan_metrics(temp_output_dir):
    """Test that NaN and Inf metrics are sanitized to None."""
    logger = TrainingProgressLogger(str(temp_output_dir))

    logger.log_metrics(
        epoch=0,
        batch=1,
        global_step=1,
        metrics={"loss": float("nan"), "accuracy": float("inf"), "valid": 1.5},
    )

    metrics_file = temp_output_dir / "graph" / "training_metrics.jsonl"
    with open(metrics_file) as f:
        data = json.loads(f.readline())
        assert data["metrics"]["loss"] is None
        assert data["metrics"]["accuracy"] is None
        assert data["metrics"]["valid"] == 1.5


def test_sanitize_negative_inf(temp_output_dir):
    """Test that negative infinity is also sanitized."""
    logger = TrainingProgressLogger(str(temp_output_dir))

    logger.log_metrics(
        epoch=0, batch=1, global_step=1, metrics={"loss": float("-inf"), "value": 2.5}
    )

    metrics_file = temp_output_dir / "graph" / "training_metrics.jsonl"
    with open(metrics_file) as f:
        data = json.loads(f.readline())
        assert data["metrics"]["loss"] is None
        assert data["metrics"]["value"] == 2.5


def test_multiple_flush_intervals(temp_output_dir):
    """Test that periodic flush resets correctly across multiple intervals."""
    logger = TrainingProgressLogger(
        str(temp_output_dir),
        flush_policy="periodic",
        flush_interval=2,
    )

    # First interval
    logger.log_metrics(epoch=0, batch=1, global_step=1, metrics={"loss": 1.0})
    assert logger._write_count == 1
    logger.log_metrics(epoch=0, batch=2, global_step=2, metrics={"loss": 0.9})
    assert logger._write_count == 0  # Reset after flush

    # Second interval
    logger.log_metrics(epoch=0, batch=3, global_step=3, metrics={"loss": 0.8})
    assert logger._write_count == 1
    logger.log_metrics(epoch=0, batch=4, global_step=4, metrics={"loss": 0.7})
    assert logger._write_count == 0  # Reset after flush


def test_default_flush_policy(temp_output_dir):
    """Test that default flush policy is immediate."""
    logger = TrainingProgressLogger(str(temp_output_dir))
    assert logger.flush_policy == "immediate"
    assert logger.flush_interval == 50


def test_custom_flush_interval(temp_output_dir):
    """Test custom flush interval configuration."""
    logger = TrainingProgressLogger(
        str(temp_output_dir),
        flush_policy="periodic",
        flush_interval=100,
    )
    assert logger.flush_interval == 100


def test_mixed_metric_types(temp_output_dir):
    """Test logging with mixed metric types (float, int, NaN)."""
    logger = TrainingProgressLogger(str(temp_output_dir))

    logger.log_metrics(
        epoch=0,
        batch=1,
        global_step=1,
        metrics={"loss": 1.5, "count": 42, "nan_value": float("nan"), "inf_value": float("inf")},
    )

    metrics_file = temp_output_dir / "graph" / "training_metrics.jsonl"
    with open(metrics_file) as f:
        data = json.loads(f.readline())
        assert data["metrics"]["loss"] == 1.5
        assert data["metrics"]["count"] == 42
        assert data["metrics"]["nan_value"] is None
        assert data["metrics"]["inf_value"] is None


def test_learning_rates_with_flush(temp_output_dir):
    """Test that learning rates are logged correctly with flush policies."""
    logger = TrainingProgressLogger(
        str(temp_output_dir),
        flush_policy="immediate",
    )

    logger.log_metrics(
        epoch=0,
        batch=1,
        global_step=1,
        metrics={"loss": 1.0},
        learning_rates={"flow_lr": 1e-4, "vae_lr": 5e-5},
    )

    metrics_file = temp_output_dir / "graph" / "training_metrics.jsonl"
    with open(metrics_file) as f:
        data = json.loads(f.readline())
        assert data["learning_rates"]["flow_lr"] == 1e-4
        assert data["learning_rates"]["vae_lr"] == 5e-5


def test_extras_with_flush(temp_output_dir):
    """Test that extras are logged correctly with flush policies."""
    logger = TrainingProgressLogger(
        str(temp_output_dir),
        flush_policy="immediate",
    )

    logger.log_metrics(
        epoch=0,
        batch=1,
        global_step=1,
        metrics={"loss": 1.0},
        extras={"batch_time": 1.23, "eta_seconds": 456},
    )

    metrics_file = temp_output_dir / "graph" / "training_metrics.jsonl"
    with open(metrics_file) as f:
        data = json.loads(f.readline())
        assert data["extras"]["batch_time"] == 1.23
        assert data["extras"]["eta_seconds"] == 456


def test_session_continuity_with_flush_policy(temp_output_dir):
    """Test that session continuity works with different flush policies."""
    # Create first logger
    logger1 = TrainingProgressLogger(str(temp_output_dir), flush_policy="immediate")
    session_id1 = logger1.session_id

    logger1.log_metrics(epoch=0, batch=1, global_step=1, metrics={"loss": 1.0})

    # Create second logger (simulating resume)
    logger2 = TrainingProgressLogger(str(temp_output_dir), flush_policy="periodic")
    session_id2 = logger2.session_id

    # Should maintain same session
    assert session_id1 == session_id2


def test_step_specific_files_with_flush(temp_output_dir):
    """Test that step-specific files work with flush policies."""
    logger = TrainingProgressLogger(
        str(temp_output_dir),
        step_name="test_step",
        flush_policy="immediate",
    )

    logger.log_metrics(epoch=0, batch=1, global_step=1, metrics={"loss": 1.0})

    # Check step-specific file exists
    metrics_file = temp_output_dir / "graph" / "training_metrics_test_step.jsonl"
    assert metrics_file.exists()
