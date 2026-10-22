"""Tests for ROCm runtime workarounds."""

import torch

from fluxflow_training.training import rocm_compat


def test_reexports_core_probe():
    from fluxflow.utils.rocm_compat import disable_broken_rocm_sdpa

    assert rocm_compat.disable_broken_rocm_sdpa is disable_broken_rocm_sdpa


def test_r1_conv_context_disables_cudnn_on_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.0")
    with rocm_compat.r1_conv_context(True):
        assert not torch.backends.cudnn.enabled
    assert torch.backends.cudnn.enabled


def test_r1_conv_context_noop_off_r1_or_without_hip(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.0")
    with rocm_compat.r1_conv_context(False):
        assert torch.backends.cudnn.enabled
    monkeypatch.setattr(torch.version, "hip", None)
    with rocm_compat.r1_conv_context(True):
        assert torch.backends.cudnn.enabled
