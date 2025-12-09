"""Integration tests for pipeline mode in train.py."""

import os
import tempfile

import pytest
import yaml

from fluxflow_training.scripts.train import (
    detect_config_mode,
    parse_args,
    validate_and_show_plan,
)


class TestDetectConfigMode:
    """Test config mode detection."""

    def test_detect_legacy_mode_no_pipeline(self):
        """Test detection of legacy mode when no pipeline section exists."""
        config = {
            "training": {
                "n_epochs": 100,
                "train_vae": True,
            }
        }
        assert detect_config_mode(config) == "legacy"

    def test_detect_legacy_mode_empty_config(self):
        """Test detection of legacy mode with empty config."""
        config = {}
        assert detect_config_mode(config) == "legacy"

    def test_detect_legacy_mode_none(self):
        """Test detection of legacy mode with None config."""
        assert detect_config_mode(None) == "legacy"

    def test_detect_pipeline_mode(self):
        """Test detection of pipeline mode when pipeline section exists."""
        config = {
            "training": {
                "pipeline": {
                    "mode": "sequential",
                    "steps": [
                        {
                            "name": "step1",
                            "duration_epochs": 10,
                            "train_vae": True,
                        }
                    ],
                }
            }
        }
        assert detect_config_mode(config) == "pipeline"


class TestValidateAndShowPlan:
    """Test pipeline validation and dry-run mode."""

    def test_validate_simple_pipeline(self, capsys):
        """Test validation of a simple valid pipeline."""
        config = {
            "training": {
                "pipeline": {
                    "mode": "sequential",
                    "steps": [
                        {
                            "name": "vae_training",
                            "description": "Train VAE",
                            "n_epochs": 50,
                            "train_vae": True,
                            "train_spade": False,
                            "freeze": ["flow_processor"],
                            "optimization": {
                                "optimizers": {
                                    "vae": {
                                        "type": "AdamW",
                                        "lr": 0.0001,
                                    }
                                },
                                "schedulers": {
                                    "vae": {
                                        "type": "CosineAnnealingLR",
                                    }
                                },
                            },
                        }
                    ],
                }
            }
        }

        # Create mock args
        class MockArgs:
            pass

        args = MockArgs()

        # Should not raise
        validate_and_show_plan(config, args)

        # Capture output
        captured = capsys.readouterr()
        assert "PIPELINE VALIDATION - DRY RUN MODE" in captured.out
        assert "Pipeline configuration is valid" in captured.out
        assert "EXECUTION PLAN" in captured.out
        assert "Step 1: vae_training" in captured.out
        assert "Duration: 50 epochs" in captured.out

    def test_validate_invalid_pipeline(self):
        """Test validation of an invalid pipeline (should raise)."""
        config = {
            "training": {
                "pipeline": {
                    "mode": "sequential",
                    "steps": [
                        {
                            "name": "invalid_step",
                            "n_epochs": 0,  # Invalid: zero epochs
                            "train_vae": False,
                            "train_diff": False,  # Invalid: no training modes
                        }
                    ],
                }
            }
        }

        class MockArgs:
            pass

        args = MockArgs()

        with pytest.raises(ValueError, match="Step 'invalid_step'"):
            validate_and_show_plan(config, args)

    def test_validate_pipeline_with_transitions(self, capsys):
        """Test validation of pipeline with transition criteria."""
        config = {
            "training": {
                "pipeline": {
                    "mode": "sequential",
                    "steps": [
                        {
                            "name": "step1",
                            "n_epochs": 100,
                            "train_vae": True,
                            "transition_on": {
                                "mode": "loss_threshold",
                                "metric": "vae_loss",
                                "threshold": 0.05,
                                "max_epochs": 100,
                            },
                        }
                    ],
                }
            }
        }

        class MockArgs:
            pass

        args = MockArgs()
        validate_and_show_plan(config, args)

        captured = capsys.readouterr()
        assert "Transition:" in captured.out
        assert "metric=vae_loss" in captured.out
        assert "max_epochs=100" in captured.out
        assert "threshold<0.05" in captured.out


class TestParseArgsIntegration:
    """Integration tests for argument parsing with config files."""

    def test_parse_args_with_pipeline_config(self):
        """Test parsing args with a pipeline config file."""
        # Create temporary config file
        config = {
            "training": {
                "pipeline": {
                    "mode": "sequential",
                    "steps": [
                        {
                            "name": "test_step",
                            "n_epochs": 10,
                            "train_vae": True,
                        }
                    ],
                }
            },
            "data": {
                "data_path": "/tmp/images",
                "captions_file": "/tmp/captions.txt",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            # Parse args with config file
            import sys

            original_argv = sys.argv
            sys.argv = ["train.py", "--config", config_path]

            args = parse_args()

            # Verify args were loaded from config
            assert args.config == config_path
            assert args.data_path == "/tmp/images"
            assert args.captions_file == "/tmp/captions.txt"

        finally:
            sys.argv = original_argv
            os.unlink(config_path)

    def test_parse_args_validate_pipeline_flag(self):
        """Test that --validate-pipeline flag is parsed correctly."""
        import sys

        original_argv = sys.argv
        sys.argv = ["train.py", "--validate-pipeline"]

        try:
            args = parse_args()
            assert args.validate_pipeline is True
        finally:
            sys.argv = original_argv


class TestEndToEndPipelineValidation:
    """End-to-end tests for pipeline validation workflow."""

    def test_validate_pipeline_end_to_end(self):
        """Test complete pipeline validation workflow."""
        # Create a complete pipeline config
        config = {
            "model": {
                "vae_dim": 128,
                "feature_maps_dim": 128,
            },
            "data": {
                "data_path": "/tmp/data",
                "captions_file": "/tmp/captions.txt",
            },
            "training": {
                "pipeline": {
                    "mode": "sequential",
                    "steps": [
                        {
                            "name": "vae_spade_off",
                            "description": "Train VAE without SPADE",
                            "n_epochs": 50,
                            "train_vae": True,
                            "train_spade": False,
                            "freeze": ["flow_processor"],
                            "optimization": {
                                "optimizers": {
                                    "vae": {
                                        "type": "AdamW",
                                        "lr": 0.0001,
                                        "betas": [0.9, 0.999],
                                        "weight_decay": 0.01,
                                    }
                                },
                                "schedulers": {
                                    "vae": {
                                        "type": "CosineAnnealingLR",
                                        "eta_min_factor": 0.01,
                                    }
                                },
                            },
                            "transition_on": {
                                "mode": "loss_threshold",
                                "metric": "vae_loss",
                                "threshold": 0.05,
                                "max_epochs": 50,
                            },
                        },
                        {
                            "name": "vae_spade_on",
                            "description": "Train VAE with SPADE",
                            "n_epochs": 50,
                            "train_vae": True,
                            "train_spade": True,
                            "freeze": ["flow_processor"],
                            "optimization": {
                                "optimizers": {
                                    "vae": {
                                        "type": "AdamW",
                                        "lr": 0.00005,
                                    }
                                },
                                "schedulers": {
                                    "vae": {
                                        "type": "CosineAnnealingLR",
                                    }
                                },
                            },
                        },
                    ],
                }
            },
        }

        # Write config to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        try:
            # Load and validate
            with open(config_path, "r") as f:
                loaded_config = yaml.safe_load(f)

            assert detect_config_mode(loaded_config) == "pipeline"

            # Validate (should not raise)
            class MockArgs:
                pass

            validate_and_show_plan(loaded_config, MockArgs())

        finally:
            os.unlink(config_path)
