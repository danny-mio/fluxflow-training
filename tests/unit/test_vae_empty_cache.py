"""Tests for memory-pressure-gated cache release in VAETrainer."""

import torch

from fluxflow_training.training import vae_trainer


def _patch_cuda(monkeypatch, free, total):
    calls = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (free, total))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    return calls


def test_cuda_cache_kept_when_memory_plentiful(monkeypatch):
    calls = _patch_cuda(monkeypatch, free=90, total=100)
    vae_trainer._empty_cache_if_low_memory(torch.device("cuda"))
    assert calls == []


def test_cuda_cache_released_when_memory_low(monkeypatch):
    calls = _patch_cuda(monkeypatch, free=5, total=100)
    vae_trainer._empty_cache_if_low_memory(torch.device("cuda"))
    assert calls == [1]


def test_cpu_is_noop(monkeypatch):
    calls = _patch_cuda(monkeypatch, free=0, total=100)
    vae_trainer._empty_cache_if_low_memory(torch.device("cpu"))
    assert calls == []
