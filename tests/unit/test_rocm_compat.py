"""Tests for ROCm runtime workarounds."""

import torch

from fluxflow_training.training import rocm_compat


def test_noop_on_cpu():
    assert rocm_compat.disable_broken_rocm_sdpa(torch.device("cpu")) == []


def test_noop_without_hip(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    assert rocm_compat.disable_broken_rocm_sdpa(torch.device("cuda")) == []


def test_disables_only_failing_backends(monkeypatch):
    state = {"EFFICIENT_ATTENTION": True, "FLASH_ATTENTION": True}
    backends = tuple(
        (name, lambda n=name: state[n], lambda v, n=name: state.__setitem__(n, v)) for name in state
    )
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(rocm_compat, "_FUSED_SDPA_BACKENDS", backends)
    monkeypatch.setattr(
        rocm_compat, "_probe_sdpa_backend", lambda name, device: name == "FLASH_ATTENTION"
    )

    disabled = rocm_compat.disable_broken_rocm_sdpa(torch.device("cuda"))

    assert disabled == ["EFFICIENT_ATTENTION"]
    assert state == {"EFFICIENT_ATTENTION": False, "FLASH_ATTENTION": True}


def test_skips_already_disabled_backends(monkeypatch):
    backends = (("EFFICIENT_ATTENTION", lambda: False, lambda v: None),)
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(rocm_compat, "_FUSED_SDPA_BACKENDS", backends)
    monkeypatch.setattr(rocm_compat, "_probe_sdpa_backend", lambda *a: False)
    assert rocm_compat.disable_broken_rocm_sdpa(torch.device("cuda")) == []


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
