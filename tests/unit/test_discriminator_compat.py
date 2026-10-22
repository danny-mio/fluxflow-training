"""Discriminator build/compat helpers (C5 spectral-norm key format, C6 ctx_dim=0 for v100)."""

import logging
from types import SimpleNamespace

import pytest
import torch.nn as nn
from fluxflow.models import PatchDiscriminator

from fluxflow_training.training.utils import (
    discriminator_ctx_dim,
    has_spectral_norm,
    load_discriminator_state,
)


def _d(sn: bool, ctx_dim: int = 0, base_ch: int = 4) -> PatchDiscriminator:
    return PatchDiscriminator(
        in_channels=3, base_ch=base_ch, depth=3, ctx_dim=ctx_dim, use_spectral_norm=sn
    )


class TestHasSpectralNorm:
    def test_detects_both_formats(self) -> None:
        assert has_spectral_norm(_d(True))
        assert not has_spectral_norm(_d(False))


class TestDiscriminatorCtxDim:
    def test_v100_compressor_gets_zero(self) -> None:
        comp = SimpleNamespace(ctx_zinject_norm=nn.Identity(), get_context_dims=lambda: 8)
        assert discriminator_ctx_dim(comp, 8) == 0

    def test_legacy_compressor_uses_context_dims(self) -> None:
        comp = SimpleNamespace(get_context_dims=lambda: 5)
        assert discriminator_ctx_dim(comp, 128) == 133

    def test_legacy_without_get_context_dims_falls_back(self) -> None:
        assert discriminator_ctx_dim(SimpleNamespace(), 128) == 133


class TestLoadDiscriminatorState:
    def test_same_format_loads_exactly(self) -> None:
        src, dst = _d(True), _d(True)
        assert load_discriminator_state(dst, src.state_dict())
        assert all(
            (a == b).all() for a, b in zip(src.state_dict().values(), dst.state_dict().values())
        )

    @pytest.mark.parametrize("saved_sn", [True, False])
    def test_spectral_norm_mismatch_rejected_and_untouched(self, saved_sn, caplog) -> None:
        saved, dst = _d(saved_sn), _d(not saved_sn)
        before = {k: v.clone() for k, v in dst.state_dict().items()}
        with caplog.at_level(logging.WARNING):
            assert not load_discriminator_state(dst, saved.state_dict())
        assert any("spectral" in r.message.lower() for r in caplog.records)
        assert all((before[k] == v).all() for k, v in dst.state_dict().items())

    def test_base_ch_mismatch_rejected(self) -> None:
        assert not load_discriminator_state(_d(True, base_ch=8), _d(True).state_dict())

    def test_legacy_ctx_proj_keys_ignored_with_info(self, caplog) -> None:
        saved, dst = _d(True, ctx_dim=16), _d(True, ctx_dim=0)
        with caplog.at_level(logging.INFO):
            assert load_discriminator_state(dst, saved.state_dict())
        assert any("ctx_proj" in r.message for r in caplog.records)
        assert (dst.head.weight_orig == saved.head.weight_orig).all()

    def test_missing_ctx_proj_when_expected_rejected(self) -> None:
        assert not load_discriminator_state(_d(True, ctx_dim=16), _d(True).state_dict())
