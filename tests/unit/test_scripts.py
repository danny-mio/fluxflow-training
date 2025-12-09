"""Unit tests for training and generation scripts."""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


def import_script_module(script_name):
    """Dynamically import a script module with mocked dependencies."""
    scripts_path = Path(__file__).parent.parent.parent / "src" / "fluxflow_training" / "scripts"
    script_path = scripts_path / f"{script_name}.py"

    # Mock heavy dependencies before import
    sys.modules["accelerate"] = Mock()
    sys.modules["torch"] = Mock()
    sys.modules["torch.nn"] = Mock()
    sys.modules["torch.optim"] = Mock()
    sys.modules["torch.optim.lr_scheduler"] = Mock()
    sys.modules["torch.utils.data"] = Mock()
    sys.modules["torchvision"] = Mock()
    sys.modules["torchvision.utils"] = Mock()
    sys.modules["transformers"] = Mock()
    sys.modules["diffusers"] = Mock()
    sys.modules["lion_pytorch"] = Mock()
    sys.modules["safetensors"] = Mock()
    sys.modules["safetensors.torch"] = Mock()

    # Mock FluxFlow modules
    sys.modules["fluxflow.models"] = Mock()
    sys.modules["fluxflow_training.data"] = Mock()
    sys.modules["fluxflow_training.training"] = Mock()
    sys.modules["fluxflow.utils"] = Mock()
    sys.modules["fluxflow_training.training.optimizer_factory"] = Mock()
    sys.modules["fluxflow_training.training.scheduler_factory"] = Mock()

    # Import the script
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    module = importlib.util.module_from_spec(spec)

    # Only execute to get the parse_args and main functions
    # We need to patch out the actual execution
    with patch.object(sys, "exit"):
        try:
            spec.loader.exec_module(module)
        except Exception:
            # Some imports might fail, but we only need parse_args
            pass

    return module


class TestTrainScriptArgumentParsing:
    """Tests for train.py argument parsing and validation."""

    def test_parse_args_minimal_tt2m(self):
        """Test parsing with minimal TTI-2M arguments."""
        train = import_script_module("train")
        test_args = [
            "--use_tt2m",
            "--tt2m_token",
            "test_token",
            "--train_vae",
            "--output_path",
            "/tmp/output",
        ]
        with patch.object(sys, "argv", ["train.py"] + test_args):
            args = train.parse_args()
            assert args.use_tt2m is True
            assert args.tt2m_token == "test_token"
            assert args.train_vae is True

    def test_parse_args_minimal_local_data(self):
        """Test parsing with local data arguments."""
        train = import_script_module("train")
        test_args = [
            "--data_path",
            "/tmp/images",
            "--captions_file",
            "/tmp/captions.tsv",
            "--train_diff",
            "--output_path",
            "/tmp/output",
        ]
        with patch.object(sys, "argv", ["train.py"] + test_args):
            args = train.parse_args()
            assert args.data_path == "/tmp/images"
            assert args.captions_file == "/tmp/captions.tsv"
            assert args.train_diff is True

    def test_parse_args_defaults(self):
        """Test default values are set correctly."""
        train = import_script_module("train")
        test_args = [
            "--data_path",
            "/tmp/images",
            "--captions_file",
            "/tmp/captions.tsv",
            "--train_vae",
            "--output_path",
            "/tmp/output",
        ]
        with patch.object(sys, "argv", ["train.py"] + test_args):
            args = train.parse_args()
            assert args.n_epochs == 1
            assert args.batch_size == 2
            assert args.workers == 1
            assert args.vae_dim == 128
            assert args.text_embedding_dim == 1024
            assert args.kl_beta == 0.0001
            assert args.kl_warmup_steps == 5000

    def test_parse_args_custom_hyperparameters(self):
        """Test custom hyperparameters are parsed."""
        train = import_script_module("train")
        test_args = [
            "--data_path",
            "/tmp/images",
            "--captions_file",
            "/tmp/captions.tsv",
            "--train_vae",
            "--output_path",
            "/tmp/output",
            "--n_epochs",
            "5",
            "--batch_size",
            "8",
            "--lr",
            "1e-5",
            "--kl_beta",
            "0.001",
            "--kl_warmup_steps",
            "10000",
            "--vae_dim",
            "256",
        ]
        with patch.object(sys, "argv", ["train.py"] + test_args):
            args = train.parse_args()
            assert args.n_epochs == 5
            assert args.batch_size == 8
            assert args.lr == 1e-5
            assert args.kl_beta == 0.001
            assert args.kl_warmup_steps == 10000
            assert args.vae_dim == 256


class TestTrainScriptOptimizerConfig:
    """Tests for optimizer/scheduler configuration loading."""

    def test_load_optimizer_config_defaults(self):
        """Test loading default optimizer configurations."""
        train = import_script_module("train")
        args = MagicMock()
        args.optim_sched_config = None

        # Mock the factory functions to return actual dictionaries
        with patch.object(
            train,
            "get_default_optimizer_config",
            side_effect=lambda name: {
                "type": "AdamW",
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "weight_decay": 0.01,
            },
        ):
            with patch.object(
                train,
                "get_default_scheduler_config",
                side_effect=lambda name: {"type": "CosineAnnealingLR", "T_max": 1000},
            ):
                lr = {"lr": 5e-7, "vae": 5e-7}
                optimizer_configs, scheduler_configs = train.load_optimizer_scheduler_config(
                    args, lr
                )

                # Check that all models have configs
                assert "flow" in optimizer_configs
                assert "vae" in optimizer_configs
                assert "text_encoder" in optimizer_configs
                assert "discriminator" in optimizer_configs

                # Check LR values are set
                assert optimizer_configs["flow"]["lr"] == 5e-7
                assert optimizer_configs["vae"]["lr"] == 5e-7

    def test_load_optimizer_config_from_file(self):
        """Test loading optimizer config from JSON file."""
        train = import_script_module("train")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "optim_config.json")
            config_data = {
                "optimizers": {
                    "flow": {
                        "type": "AdamW",
                        "lr": 1e-4,
                        "betas": [0.9, 0.999],
                        "weight_decay": 0.01,
                    }
                },
                "schedulers": {"flow": {"type": "CosineAnnealingLR", "T_max": 1000}},
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            args = MagicMock()
            args.optim_sched_config = config_path

            # Mock the factory functions to return actual dictionaries
            with patch.object(
                train,
                "get_default_optimizer_config",
                side_effect=lambda name: {
                    "type": "AdamW",
                    "lr": 1e-4,
                    "betas": [0.9, 0.999],
                    "weight_decay": 0.01,
                },
            ):
                with patch.object(
                    train,
                    "get_default_scheduler_config",
                    side_effect=lambda name: {"type": "CosineAnnealingLR", "T_max": 1000},
                ):
                    lr = {"lr": 5e-7, "vae": 5e-7}
                    optimizer_configs, scheduler_configs = train.load_optimizer_scheduler_config(
                        args, lr
                    )

                    # Check loaded config is used
                    assert optimizer_configs["flow"]["type"] == "AdamW"
                    assert (
                        optimizer_configs["flow"]["lr"] == 1e-4
                    )  # Config file lr is used when specified
                    assert scheduler_configs["flow"]["type"] == "CosineAnnealingLR"

    def test_load_optimizer_config_missing_file(self):
        """Test handling of missing config file."""
        train = import_script_module("train")
        args = MagicMock()
        args.optim_sched_config = "/nonexistent/path.json"

        # Mock the factory functions to return actual dictionaries
        with patch.object(
            train,
            "get_default_optimizer_config",
            side_effect=lambda name: {
                "type": "AdamW",
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "weight_decay": 0.01,
            },
        ):
            with patch.object(
                train,
                "get_default_scheduler_config",
                side_effect=lambda name: {"type": "CosineAnnealingLR", "T_max": 1000},
            ):
                lr = {"lr": 5e-7, "vae": 5e-7}
                # Should fall back to defaults without error
                optimizer_configs, scheduler_configs = train.load_optimizer_scheduler_config(
                    args, lr
                )

                assert "flow" in optimizer_configs
                assert "vae" in optimizer_configs


class TestGenerateScriptArgumentParsing:
    """Tests for generate.py argument parsing."""

    def test_parse_args_minimal(self):
        """Test parsing with minimal required arguments."""
        generate = import_script_module("generate")
        test_args = [
            "--model_checkpoint",
            "/tmp/model.safetensors",
            "--text_prompts_path",
            "/tmp/prompts",
        ]
        with patch.object(sys, "argv", ["generate.py"] + test_args):
            args = generate.parse_args()
            assert args.model_checkpoint == "/tmp/model.safetensors"
            assert args.text_prompts_path == "/tmp/prompts"


class TestTrainScriptMain:
    """Tests for main() entry point validation."""

    def test_main_validates_data_args(self, capsys):
        """Test that main() validates data arguments."""
        train = import_script_module("train")
        test_args = ["--train_vae", "--output_path", "/tmp/output"]  # Training mode but no data
        with patch.object(sys, "argv", ["train.py"] + test_args):
            with pytest.raises(SystemExit) as exc_info:
                train.main()
            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Error:" in captured.out or "error" in captured.out.lower()

    def test_main_warns_no_training_mode(self, capsys):
        """Test warning when no training mode is enabled."""
        train = import_script_module("train")
        test_args = [
            "--data_path",
            "/tmp/images",
            "--captions_file",
            "/tmp/captions.tsv",
            "--output_path",
            "/tmp/output",
            # No --train_vae or --train_diff
        ]
        with patch.object(sys, "argv", ["train.py"] + test_args):
            # Mock the train_legacy function to avoid actually running training
            with patch.object(train, "train_legacy", return_value=None):
                train.main()

        captured = capsys.readouterr()
        assert "Warning:" in captured.out or "warning" in captured.out.lower()
