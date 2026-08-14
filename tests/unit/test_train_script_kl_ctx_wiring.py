"""Unit tests: legacy scripts/train.py must wire kl_z_weight / ctx_shrinkage_weight
into VAETrainer(...), matching the discriminator_update_freq / lambda_adv pattern.

Source-inspection tests (matches TestTrainReconstructionParameter /
TestKlZWeightAndCtxShrinkageWiring style in test_pipeline_orchestrator.py) since
scripts/train.py's CLI + argparse plumbing is impractical to exercise end-to-end.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def train_script_content() -> str:
    train_path = (
        Path(__file__).parent.parent.parent / "src" / "fluxflow_training" / "scripts" / "train.py"
    )
    return train_path.read_text()


class TestKlZWeightCliArg:
    def test_kl_z_weight_arg_defined(self, train_script_content: str) -> None:
        assert '"--kl_z_weight"' in train_script_content

    def test_kl_z_weight_default_is_zero(self, train_script_content: str) -> None:
        # Default must match VAETrainer's own inert default (0.0) so the legacy
        # driver's behavior is unchanged unless a user explicitly opts in.
        idx = train_script_content.index('"--kl_z_weight"')
        snippet = train_script_content[idx : idx + 200]
        assert "default=0.0" in snippet

    def test_kl_z_weight_passed_to_vae_trainer(self, train_script_content: str) -> None:
        assert "kl_z_weight=args.kl_z_weight" in train_script_content

    def test_kl_z_weight_yaml_override(self, train_script_content: str) -> None:
        assert '"kl_z_weight" in config["training"] and "kl_z_weight" not in cli_provided' in (
            train_script_content
        )
        assert 'args.kl_z_weight = config["training"]["kl_z_weight"]' in train_script_content


class TestCtxShrinkageWeightCliArg:
    def test_ctx_shrinkage_weight_arg_defined(self, train_script_content: str) -> None:
        assert '"--ctx_shrinkage_weight"' in train_script_content

    def test_ctx_shrinkage_weight_default_is_zero(self, train_script_content: str) -> None:
        idx = train_script_content.index('"--ctx_shrinkage_weight"')
        snippet = train_script_content[idx : idx + 200]
        assert "default=0.0" in snippet

    def test_ctx_shrinkage_weight_passed_to_vae_trainer(self, train_script_content: str) -> None:
        assert "ctx_shrinkage_weight=args.ctx_shrinkage_weight" in train_script_content

    def test_ctx_shrinkage_weight_yaml_override(self, train_script_content: str) -> None:
        assert '"ctx_shrinkage_weight" in config["training"]' in train_script_content
        assert '"ctx_shrinkage_weight" not in cli_provided' in train_script_content
        assert (
            'args.ctx_shrinkage_weight = config["training"]["ctx_shrinkage_weight"]'
            in train_script_content
        )
