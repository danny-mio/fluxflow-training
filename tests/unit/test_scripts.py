"""Unit tests for training and generation scripts."""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

_MOCKED_MODULES = [
    "accelerate",
    "torch",
    "torch.nn",
    "torch.optim",
    "torch.optim.lr_scheduler",
    "torch.utils.data",
    "torchvision",
    "torchvision.utils",
    "transformers",
    "diffusers",
    "lion_pytorch",
    "safetensors",
    "safetensors.torch",
    "fluxflow.models",
    "fluxflow_training.data",
    "fluxflow_training.training",
    "fluxflow.utils",
    "fluxflow_training.training.optimizer_factory",
    "fluxflow_training.training.scheduler_factory",
]


def import_script_module(script_name):
    """Dynamically import a script module with mocked dependencies.

    Uses ``patch.dict`` (scoped to this call) rather than mutating
    ``sys.modules`` directly, so the mocks don't leak into later tests in the
    same pytest session -- a real torch/transformers import elsewhere (e.g.
    ``tests/unit/test_datasets.py``) would otherwise silently pick up these
    Mock() stand-ins if it happened to run afterward.
    """
    scripts_path = Path(__file__).parent.parent.parent / "src" / "fluxflow_training" / "scripts"
    script_path = scripts_path / f"{script_name}.py"

    with patch.dict(sys.modules, {name: Mock() for name in _MOCKED_MODULES}):
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
            assert args.max_text_length == 32
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

    def test_parse_args_max_text_length_override(self):
        """--max_text_length overrides the 32-token v0.10.0 default."""
        train = import_script_module("train")
        test_args = [
            "--data_path",
            "/tmp/images",
            "--captions_file",
            "/tmp/captions.tsv",
            "--train_vae",
            "--output_path",
            "/tmp/output",
            "--max_text_length",
            "64",
        ]
        with patch.object(sys, "argv", ["train.py"] + test_args):
            args = train.parse_args()
            assert args.max_text_length == 64

    def test_parse_args_precision_defaults_to_fp16(self):
        """--precision defaults to 'fp16' when not passed (bf16 mitigation opt-in only)."""
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
            assert args.precision == "fp16"
            assert args.use_fp16 is False

    def test_parse_args_precision_bf16_cli(self):
        """--precision bf16 is accepted on the CLI alongside --use_fp16."""
        train = import_script_module("train")
        test_args = [
            "--data_path",
            "/tmp/images",
            "--captions_file",
            "/tmp/captions.tsv",
            "--train_vae",
            "--output_path",
            "/tmp/output",
            "--use_fp16",
            "--precision",
            "bf16",
        ]
        with patch.object(sys, "argv", ["train.py"] + test_args):
            args = train.parse_args()
            assert args.use_fp16 is True
            assert args.precision == "bf16"

    def test_parse_args_precision_rejects_invalid_choice(self):
        """argparse ``choices`` rejects any value outside {fp16, bf16}."""
        train = import_script_module("train")
        test_args = [
            "--data_path",
            "/tmp/images",
            "--captions_file",
            "/tmp/captions.tsv",
            "--train_vae",
            "--output_path",
            "/tmp/output",
            "--precision",
            "fp32",
        ]
        with patch.object(sys, "argv", ["train.py"] + test_args):
            with pytest.raises(SystemExit):
                train.parse_args()


class TestTrainScriptConfigFileMerge:
    """Tests for YAML config merging into CLI args (train.parse_args)."""

    def test_config_max_text_length_applied_when_not_on_cli(self, tmp_path):
        train = import_script_module("train")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "data:\n"
            "  data_path: /tmp/images\n"
            "  captions_file: /tmp/captions.tsv\n"
            "  max_text_length: 96\n"
            "training:\n"
            "  train_vae: true\n"
        )
        test_args = ["--config", str(config_path), "--output_path", "/tmp/output"]
        with patch.object(sys, "argv", ["train.py"] + test_args):
            args = train.parse_args()
            assert args.max_text_length == 96

    def test_cli_max_text_length_overrides_config(self, tmp_path):
        train = import_script_module("train")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "data:\n"
            "  data_path: /tmp/images\n"
            "  captions_file: /tmp/captions.tsv\n"
            "  max_text_length: 96\n"
            "training:\n"
            "  train_vae: true\n"
        )
        test_args = [
            "--config",
            str(config_path),
            "--output_path",
            "/tmp/output",
            "--max_text_length",
            "48",
        ]
        with patch.object(sys, "argv", ["train.py"] + test_args):
            args = train.parse_args()
            assert args.max_text_length == 48

    def test_config_use_fp16_only_still_defaults_precision_to_fp16(self, tmp_path):
        """Backward compat: a config with only ``use_fp16: true`` (no ``precision``
        key at all) must keep resolving to fp16, unchanged from pre-bf16 behavior."""
        train = import_script_module("train")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "data:\n"
            "  data_path: /tmp/images\n"
            "  captions_file: /tmp/captions.tsv\n"
            "training:\n"
            "  train_vae: true\n"
            "  use_fp16: true\n"
        )
        test_args = ["--config", str(config_path), "--output_path", "/tmp/output"]
        with patch.object(sys, "argv", ["train.py"] + test_args):
            args = train.parse_args()
            assert args.use_fp16 is True
            assert args.precision == "fp16"

    def test_config_precision_bf16_applied_when_not_on_cli(self, tmp_path):
        """``training.precision: bf16`` in YAML is picked up when not passed on the CLI."""
        train = import_script_module("train")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "data:\n"
            "  data_path: /tmp/images\n"
            "  captions_file: /tmp/captions.tsv\n"
            "training:\n"
            "  train_vae: true\n"
            "  use_fp16: true\n"
            "  precision: bf16\n"
        )
        test_args = ["--config", str(config_path), "--output_path", "/tmp/output"]
        with patch.object(sys, "argv", ["train.py"] + test_args):
            args = train.parse_args()
            assert args.use_fp16 is True
            assert args.precision == "bf16"

    def test_cli_precision_overrides_config(self, tmp_path):
        """--precision on the CLI takes priority over ``training.precision`` in YAML."""
        train = import_script_module("train")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "data:\n"
            "  data_path: /tmp/images\n"
            "  captions_file: /tmp/captions.tsv\n"
            "training:\n"
            "  train_vae: true\n"
            "  use_fp16: true\n"
            "  precision: bf16\n"
        )
        test_args = [
            "--config",
            str(config_path),
            "--output_path",
            "/tmp/output",
            "--precision",
            "fp16",
        ]
        with patch.object(sys, "argv", ["train.py"] + test_args):
            args = train.parse_args()
            assert args.precision == "fp16"


class TestResolveMixedPrecision:
    """Tests for ``train._resolve_mixed_precision`` — the single choke point all four
    ``Accelerator(...)`` construction call sites go through to pick ``mixed_precision``.

    fp16's narrow dynamic range clips growing activations to Inf; bf16 shares fp32's
    exponent range (no overflow-to-Inf) at the cost of mantissa precision, which is an
    acceptable tradeoff for the fp16 NaN/Inf training crash this flag mitigates.
    """

    def test_use_fp16_false_ignores_precision(self):
        """``use_fp16=False`` must always resolve to 'no', regardless of ``precision``."""
        train = import_script_module("train")
        assert train._resolve_mixed_precision(False, "bf16") == "no"
        assert train._resolve_mixed_precision(False, "fp16") == "no"

    def test_use_fp16_true_default_precision_is_fp16(self):
        """Unchanged historical behavior: ``use_fp16=True`` + default precision -> fp16."""
        train = import_script_module("train")
        assert train._resolve_mixed_precision(True, "fp16") == "fp16"

    def test_use_fp16_true_bf16_precision(self):
        """``use_fp16=True`` + ``precision='bf16'`` -> bf16 (the new mitigation path)."""
        train = import_script_module("train")
        assert train._resolve_mixed_precision(True, "bf16") == "bf16"

    def test_invalid_precision_raises(self):
        """Any value outside {fp16, bf16} raises, even if it slipped in via YAML
        (which bypasses argparse's ``choices`` validation)."""
        train = import_script_module("train")
        with pytest.raises(ValueError):
            train._resolve_mixed_precision(True, "fp32")


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


class _FakeToModule:
    """Minimal stand-in for an nn.Module: supports the handful of calls
    initialize_models() makes on compressor/expander/flow_processor/text_encoder/D_img
    in the factory (use_factory=True) path."""

    def to(self, device):
        return self

    def named_parameters(self):
        return []

    def float(self):
        return self


class _FakeCompressor(_FakeToModule):
    def get_context_dims(self):
        return 5


class TestInitializeModelsAttentionBackendPrecedence:
    """initialize_models() builds ModelConfig for the factory (model_type-driven)
    path directly from the raw YAML config['model'] dict, bypassing the
    already-merged, CLI-precedence-correct `args.attention_backend`. This means a
    user passing both --config some.yaml and --attention_backend sdpa has their CLI
    choice silently discarded whenever --config selects the factory path (any
    config['model'] with a model_type key) -- see train.parse_args()'s cli_provided
    merge (~line 1995) which computes the correct value into args, but
    initialize_models() (~line 227) never reads it.
    """

    def _call_initialize_models(self, train, *, cli_attention_backend, yaml_attention_backend):
        config = {"model": {"model_type": "bezier"}}
        if yaml_attention_backend is not None:
            config["model"]["attention_backend"] = yaml_attention_backend

        args = MagicMock()
        args.channels = 3
        args.use_gradient_checkpointing = False
        args.text_embedding_dim = 1024
        args.feature_maps_dim_disc = 8
        args.vae_dim = 128
        args.model_checkpoint = None
        args.output_path = None
        args.attention_backend = cli_attention_backend

        fake_models = (_FakeCompressor(), _FakeToModule(), _FakeToModule(), _FakeToModule())
        captured = {}

        def fake_create_models_from_config(model_config):
            captured["model_config"] = model_config
            return fake_models

        with patch.object(
            train, "create_models_from_config", side_effect=fake_create_models_from_config
        ):
            with patch.object(train, "PatchDiscriminator", return_value=_FakeToModule()):
                train.initialize_models(args, config, device="cpu", checkpoint_manager=MagicMock())

        return captured["model_config"]

    def test_cli_attention_backend_overrides_yaml_value(self):
        """CLI --attention_backend sdpa must win even though the YAML config
        explicitly sets attention_backend: einsum (bug: it was silently ignored)."""
        train = import_script_module("train")
        model_config = self._call_initialize_models(
            train, cli_attention_backend="sdpa", yaml_attention_backend="einsum"
        )
        assert model_config.attention_backend == "sdpa"

    def test_cli_attention_backend_used_when_yaml_omits_it(self):
        """No attention_backend key in YAML -> the (merged) args value is used,
        not ModelConfig's own default."""
        train = import_script_module("train")
        model_config = self._call_initialize_models(
            train, cli_attention_backend="einsum", yaml_attention_backend=None
        )
        assert model_config.attention_backend == "einsum"
