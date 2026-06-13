"""Unit tests for v0.10.0 CFG empty-prompt substitution.

The substitution form is the per-token-text replacement for the legacy
zero-out (``apply_cfg_dropout``). It must:
- Be a no-op when p_uncond == 0.
- Substitute roughly the right fraction of samples when p_uncond > 0.
- Preserve shapes.
- Cache the encoded null pair across calls (we shouldn't re-tokenize every step).
- Round-trip through both 3D per-token text_seq and (sane) 2D edge cases.
"""

import torch
import torch.nn as nn

from fluxflow_training.training.cfg_utils import apply_cfg_null_substitution


class _MockTextEncoder(nn.Module):
    """Mock encoder that returns deterministic (text_seq, text_mask) tuples.

    For null calls (recognizable by input shape via tokenizer side) it returns
    a constant null pattern. For real calls it returns ones * scale.
    """

    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        # Need at least one parameter so .parameters() returns a device.
        self.dummy = nn.Linear(1, 1)
        self.call_count = 0
        self.last_input_ids_shape = None

    def forward(self, input_ids, attention_mask=None):
        self.call_count += 1
        B, T = input_ids.shape
        # Null vector is a constant 99.0 so test can recognize substitution.
        text_seq = torch.full((B, T, self.embed_dim), 99.0)
        if attention_mask is not None:
            text_mask = attention_mask.bool()
        else:
            text_mask = torch.ones(B, T, dtype=torch.bool)
        return text_seq, text_mask


class TestApplyCfgNullSubstitution:
    def _make_inputs(self, B: int = 8, T: int = 32, E: int = 8):
        # Real text: all 1.0s, all-valid mask.
        text_seq = torch.ones(B, T, E)
        text_mask = torch.ones(B, T, dtype=torch.bool)
        return text_seq, text_mask

    def test_no_op_when_p_zero(self):
        text_seq, text_mask = self._make_inputs()
        enc = _MockTextEncoder()
        out_seq, out_mask = apply_cfg_null_substitution(
            text_seq, text_mask, text_encoder=enc, p_uncond=0.0
        )
        assert torch.equal(out_seq, text_seq)
        assert torch.equal(out_mask, text_mask)
        # Encoder should not be called when CFG is disabled.
        assert enc.call_count == 0

    def test_full_substitution_when_p_one(self):
        text_seq, text_mask = self._make_inputs(B=4, T=8, E=8)
        enc = _MockTextEncoder(embed_dim=8)
        torch.manual_seed(0)
        out_seq, _ = apply_cfg_null_substitution(
            text_seq, text_mask, text_encoder=enc, p_uncond=1.0
        )
        # All samples should now be the null (=99.0).
        assert torch.all(out_seq == 99.0)

    def test_shapes_preserved(self):
        text_seq, text_mask = self._make_inputs(B=6, T=12, E=4)
        enc = _MockTextEncoder(embed_dim=4)
        out_seq, out_mask = apply_cfg_null_substitution(
            text_seq, text_mask, text_encoder=enc, p_uncond=0.5
        )
        assert out_seq.shape == text_seq.shape
        assert out_mask.shape == text_mask.shape

    def test_null_pair_cached_across_calls(self):
        """The null pair should be built once and reused. Encoder gets called
        at most once for the null prompt across many CFG-dropout invocations."""
        text_seq, text_mask = self._make_inputs(B=4, T=16, E=8)
        enc = _MockTextEncoder(embed_dim=8)
        torch.manual_seed(0)

        for _ in range(5):
            apply_cfg_null_substitution(text_seq, text_mask, text_encoder=enc, p_uncond=0.5)
        # Encoder should have been called exactly once (to build the cached null).
        assert enc.call_count == 1
        assert hasattr(enc, "_cfg_null_pair")

    def test_partial_substitution_fraction_matches_p(self):
        """At p=0.5 over many samples, ~half should be null."""
        text_seq, text_mask = self._make_inputs(B=1000, T=8, E=4)
        enc = _MockTextEncoder(embed_dim=4)
        torch.manual_seed(42)
        out_seq, _ = apply_cfg_null_substitution(
            text_seq, text_mask, text_encoder=enc, p_uncond=0.5
        )
        # A row is null iff its first embed value is 99.0
        null_rows = (out_seq[:, 0, 0] == 99.0).sum().item()
        # ~500 ± 50 for 1000 trials at p=0.5
        assert 400 <= null_rows <= 600

    def test_invalid_p_raises(self):
        text_seq, text_mask = self._make_inputs()
        enc = _MockTextEncoder()
        import pytest

        with pytest.raises(ValueError):
            apply_cfg_null_substitution(text_seq, text_mask, text_encoder=enc, p_uncond=1.5)
        with pytest.raises(ValueError):
            apply_cfg_null_substitution(text_seq, text_mask, text_encoder=enc, p_uncond=-0.01)

    def test_does_not_mutate_inputs(self):
        """The function should return new tensors; the original real text must
        not be clobbered (caller may still want to inspect/log it)."""
        text_seq, text_mask = self._make_inputs(B=4, T=8, E=4)
        text_seq_before = text_seq.clone()
        text_mask_before = text_mask.clone()
        enc = _MockTextEncoder(embed_dim=4)
        torch.manual_seed(0)
        apply_cfg_null_substitution(text_seq, text_mask, text_encoder=enc, p_uncond=0.9)
        # Originals must be unchanged
        assert torch.equal(text_seq, text_seq_before)
        assert torch.equal(text_mask, text_mask_before)
