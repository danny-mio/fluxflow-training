"""Unit tests: scripts/train.py's legacy resume path must warn when a loaded
flow_processor checkpoint predates fluxflow-core's dual-FiLM identity-at-init
fix (Plan02 Fix B). Pre-fix checkpoints lack film_text_scale/film_time_scale;
loading with strict=False silently defaults both to 0, which completely
zeroes (not merely attenuates) all timestep/text FiLM conditioning
model-wide. See plans/02-flow-v0.10.0-improvements.md, Fix B, mitigation (b).

Combines real behavioral unit tests of the extracted helper
(_warn_if_film_scales_missing) with one source-inspection test confirming
the helper is actually wired into the resume block, matching the wiring
style in test_train_script_kl_ctx_wiring.py.
"""

from pathlib import Path

import pytest
import torch

from fluxflow_training.scripts.train import _warn_if_film_scales_missing


@pytest.fixture(scope="module")
def train_script_content() -> str:
    train_path = (
        Path(__file__).parent.parent.parent / "src" / "fluxflow_training" / "scripts" / "train.py"
    )
    return train_path.read_text()


class TestWarnIfFilmScalesMissing:
    def test_both_scales_present_top_level_no_warning(self, capsys: pytest.CaptureFixture) -> None:
        state_dict = {
            "film_text_scale": torch.zeros(1),
            "film_time_scale": torch.zeros(1),
            "some_other_param": torch.zeros(4),
        }
        warned = _warn_if_film_scales_missing(state_dict)
        assert warned is False
        assert capsys.readouterr().out == ""

    def test_both_scales_present_per_block_keys_no_warning(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        # Real checkpoints hold these under nn.ModuleList block prefixes,
        # e.g. "transformer_blocks.3.film_text_scale".
        state_dict = {
            f"transformer_blocks.{i}.film_text_scale": torch.zeros(1) for i in range(10)
        } | {f"transformer_blocks.{i}.film_time_scale": torch.zeros(1) for i in range(10)}
        warned = _warn_if_film_scales_missing(state_dict)
        assert warned is False
        assert capsys.readouterr().out == ""

    def test_both_scales_missing_warns(self, capsys: pytest.CaptureFixture) -> None:
        # Simulates a pre-Fix-B checkpoint: no film_text_scale/film_time_scale
        # keys at all.
        state_dict = {
            "transformer_blocks.0.norm1.weight": torch.zeros(4),
            "transformer_blocks.0.film_p0_text.weight": torch.zeros(4, 4),
        }
        warned = _warn_if_film_scales_missing(state_dict)
        assert warned is True
        out = capsys.readouterr().out
        assert "film_text_scale" in out
        assert "film_time_scale" in out
        assert "reset to zero" in out
        assert "fine-tuning" in out

    def test_only_time_scale_missing_still_warns(self, capsys: pytest.CaptureFixture) -> None:
        state_dict = {
            "transformer_blocks.0.film_text_scale": torch.zeros(1),
        }
        warned = _warn_if_film_scales_missing(state_dict)
        assert warned is True
        assert "film_time_scale" in capsys.readouterr().out

    def test_only_text_scale_missing_still_warns(self, capsys: pytest.CaptureFixture) -> None:
        state_dict = {
            "transformer_blocks.0.film_time_scale": torch.zeros(1),
        }
        warned = _warn_if_film_scales_missing(state_dict)
        assert warned is True
        assert "film_text_scale" in capsys.readouterr().out

    def test_empty_state_dict_warns(self, capsys: pytest.CaptureFixture) -> None:
        warned = _warn_if_film_scales_missing({})
        assert warned is True
        out = capsys.readouterr().out
        assert "film_text_scale" in out
        assert "film_time_scale" in out


class TestFilmScaleWarningWiredIntoResume:
    """Wiring test: mirrors the style of test_train_script_kl_ctx_wiring.py.

    Confirms _warn_if_film_scales_missing is actually called from the
    train_legacy resume block, right after the flow_processor.load_state_dict
    call -- not just that the helper exists in isolation.
    """

    def test_helper_called_after_flow_processor_load(self, train_script_content: str) -> None:
        # train.py has two flow_processor.load_state_dict(..., strict=False)
        # call sites: initialize_models() (pipeline mode) and train_legacy()'s
        # resume block. Only the latter is in scope (plan Fix B mitigation
        # (b) cites train_legacy's resume logic specifically) -- scope the
        # search to inside def train_legacy(...) so a match in
        # initialize_models() can't false-positive this test.
        train_legacy_start = train_script_content.index("def train_legacy(")
        anchor = (
            'flow_processor.load_state_dict(loaded_states["diffuser.flow_processor"], '
            "strict=False)"
        )
        idx = train_script_content.index(anchor, train_legacy_start)
        snippet = train_script_content[idx : idx + 400]
        assert "_warn_if_film_scales_missing(" in snippet

    def test_helper_receives_checkpoint_state_dict_not_model_state_dict(
        self, train_script_content: str
    ) -> None:
        # Must be called with the *loaded* checkpoint dict (which lacks the
        # keys pre-fix), not flow_processor.state_dict() (which always has
        # them, defaulted to 0 by nn.Parameter init -- that would never
        # trigger the warning).
        train_legacy_start = train_script_content.index("def train_legacy(")
        anchor = "_warn_if_film_scales_missing("
        idx = train_script_content.index(anchor, train_legacy_start)
        snippet = train_script_content[idx : idx + 100]
        assert 'loaded_states["diffuser.flow_processor"]' in snippet

    def test_helper_not_called_in_initialize_models_pipeline_path(
        self, train_script_content: str
    ) -> None:
        # Documents current scope: the pipeline-mode resume path in
        # initialize_models() has the identical gap but is out of scope for
        # this fix (plan Fix B mitigation (b) only covers train_legacy's
        # resume logic). If this starts failing because someone wires the
        # helper into initialize_models() too, that's a welcome expansion of
        # coverage -- update/remove this test rather than treating it as a
        # regression.
        start = train_script_content.index("def initialize_models(")
        end = train_script_content.index("\ndef ", start + 1)
        initialize_models_body = train_script_content[start:end]
        assert "_warn_if_film_scales_missing(" not in initialize_models_body
