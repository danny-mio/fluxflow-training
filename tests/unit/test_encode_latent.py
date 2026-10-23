"""encode_latent: pass `sample` only to compressors that accept it (v100); legacy untouched."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from fluxflow_training.training.utils import encode_latent


class _V100(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, img, training=False, sample=None):
        self.seen = sample
        return img


class _Legacy(nn.Module):
    def forward(self, img):
        return img


class _Kwargs(nn.Module):
    def forward(self, img, **kw):
        self.kw = kw
        return img


@pytest.mark.parametrize("sample", [True, False])
def test_passes_sample_to_v100(sample):
    m = _V100()
    encode_latent(m, torch.zeros(1), sample=sample)
    assert m.seen is sample


def test_legacy_without_sample_param():
    x = torch.ones(2)
    assert torch.equal(encode_latent(_Legacy(), x, sample=True), x)


def test_var_kwargs_not_assumed_to_accept_sample():
    m = _Kwargs()
    encode_latent(m, torch.zeros(1), sample=True)
    assert m.kw == {}


def test_uninspectable_callable_falls_back():
    m = MagicMock(return_value="out", spec=[])
    m.forward = 5  # not inspectable
    assert encode_latent(m, "x", sample=True) == "out"
    m.assert_called_once_with("x")


def test_typeerror_inside_forward_propagates():
    class Bad(nn.Module):
        def forward(self, img, sample=None):
            raise TypeError("boom")

    with pytest.raises(TypeError, match="boom"):
        encode_latent(Bad(), torch.zeros(1), sample=True)
