"""Audit fixes for VAETrainer: adaptive GAN weight, GAN warmup, schedulers, D accumulation, fp16."""

import inspect
import logging
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fluxflow_training.training.vae_trainer import VAETrainer
from tests.unit.test_vae_trainer_fp16_grad_scaler import _FakeAccelerator
from tests.unit.test_vae_trainer_gradient_accumulation import (
    _IMG_SHAPE,
    _TOKEN_DIM,
    _FakeCompressor,
    _FakeDiscriminator,
    _FakeExpander,
    _PlainAccelerator,
)


class _RgbExpander(_FakeExpander):
    """Expander with a named last layer (to_rgb_conv) like FluxExpander_v100."""

    def __init__(self, img_shape):
        super().__init__(img_shape)
        self.to_rgb_conv = nn.Sequential(nn.Conv2d(3, 3, 1), nn.Conv2d(3, 3, 1))

    def forward(self, packed, use_context=True):
        return self.to_rgb_conv(super().forward(packed, use_context))


def _imgs(b: int = 2) -> torch.Tensor:
    return torch.randn(b, *_IMG_SHAPE)


def _mk(gan=False, rgb=False, accel=None, d_ctx_dim=_TOKEN_DIM, **over) -> VAETrainer:
    comp = _FakeCompressor(_IMG_SHAPE[0], _TOKEN_DIM)
    exp = (_RgbExpander if rgb else _FakeExpander)(_IMG_SHAPE)
    opt = torch.optim.SGD(list(comp.parameters()) + list(exp.parameters()), lr=1e-3)
    kw = dict(
        compressor=comp,
        expander=exp,
        optimizer=opt,
        scheduler=torch.optim.lr_scheduler.StepLR(opt, step_size=1),
        ema=MagicMock(),
        reconstruction_loss_fn=nn.L1Loss(),
        reconstruction_loss_min_fn=nn.MSELoss(),
        train_kl=False,
        train_colorstats=False,
        train_histogram=False,
        train_contrast=False,
        train_coarseness=False,
        use_lpips=False,
        ctx_input_dim=_TOKEN_DIM,
        context_channels=2,
        context_height=2,
        context_width=2,
        r1_interval=1000,
        instance_noise_std=0.0,
        accelerator=accel or _PlainAccelerator(),
        use_gan=gan,
        gan_warmup_steps=0,
    )
    if gan:
        d = _FakeDiscriminator(ctx_dim=d_ctx_dim)
        d_opt = torch.optim.SGD(d.parameters(), lr=1e-3)
        kw.update(
            discriminator=d,
            discriminator_optimizer=d_opt,
            discriminator_scheduler=torch.optim.lr_scheduler.StepLR(d_opt, step_size=1),
        )
    kw.update(over)
    return VAETrainer(**kw)


def _count(obj, name):
    calls = []
    orig = getattr(obj, name)
    setattr(obj, name, lambda *a, **k: calls.append(1) or orig(*a, **k))
    return calls


class TestContract:
    def test_new_kwargs_and_defaults(self):
        p = inspect.signature(VAETrainer.__init__).parameters
        assert "total_training_steps" not in p
        assert p["gan_start_step"].default == 0
        assert p["gan_warmup_steps"].default == 1000
        assert p["train_ctx_aux"].default is False
        assert p["instance_noise_std"].default == 0.05

    def test_loss_history_weighting_removed(self):
        t = _mk()
        assert not hasattr(t, "loss_history")
        assert not hasattr(t, "_compute_adaptive_weight")


class TestAdaptiveGanWeight:
    def test_ratio_formula(self):
        w = nn.Parameter(torch.ones(2))
        t = _mk()
        out = t._adaptive_gan_weight((w * 3).sum(), (w * 1).sum(), w)
        assert out == pytest.approx(3 * 2**0.5 / (2**0.5 + 1e-4), rel=1e-5)

    def test_clamped_to_1e4(self):
        w = nn.Parameter(torch.ones(2))
        t = _mk()
        out = t._adaptive_gan_weight((w * 3).sum(), (w * 0).sum(), w)
        assert out == 1e4

    def test_no_grad_path_falls_back_with_one_warning(self, caplog):
        w = nn.Parameter(torch.ones(2))
        other = nn.Parameter(torch.ones(2))
        t = _mk()
        with caplog.at_level(logging.WARNING):
            a = t._adaptive_gan_weight((w * 3).sum(), (other * 1).sum(), w)
            b = t._adaptive_gan_weight((w * 3).sum(), (other * 1).sum(), w)
        assert a is None and b is None
        assert sum("adaptive" in r.message.lower() for r in caplog.records) == 1

    def test_weight_never_negative_with_negative_g_loss(self):
        t = _mk(gan=True, rgb=True, lambda_adv=0.5)
        t.discriminator.conv.bias.data.fill_(50.0)  # G hinge strongly negative
        r = t._train_generator(_imgs(), global_step=1)
        assert r["generator"] < 0
        assert 0.0 <= r["weight_gan"] <= 0.5 * 1e4

    def test_adaptive_differs_from_fixed_and_fixed_is_lambda(self):
        fixed = _mk(gan=True, rgb=True, adaptive_weights=False, lambda_adv=0.5)
        assert fixed._train_generator(_imgs(), 1)["weight_gan"] == pytest.approx(0.5)
        ad = _mk(gan=True, rgb=True, adaptive_weights=True, lambda_adv=0.5)
        w = ad._train_generator(_imgs(), 1)["weight_gan"]
        assert w != pytest.approx(0.5) and w > 0

    def test_fallback_without_last_layer_uses_fixed(self):
        t = _mk(gan=True, rgb=False, lambda_adv=0.5)
        assert t._train_generator(_imgs(), 1)["weight_gan"] == pytest.approx(0.5)

    def test_kl_never_weighted(self):
        t = _mk(gan=True, rgb=True, train_kl=True, kl_z_weight=0.5, kl_z_warmup_steps=0)
        captured = {}
        orig = t.accelerator.backward
        t.accelerator.backward = lambda loss: captured.update(loss=loss.item()) or orig(loss)
        mu = torch.full((2, 4, 2, 2), 2.0)
        t.compressor.forward = lambda x, training=False: (
            _FakeCompressor.forward(t.compressor, x)[0],
            mu,
            torch.zeros_like(mu),
        )
        r = t._train_generator(_imgs(), 1)
        assert r["kl"] == pytest.approx(2.0)  # 0.5*mu^2 per dim
        t2 = _mk(train_kl=False)
        assert t2._train_generator(_imgs(), 1)["kl"] == 0.0


class TestGanWarmup:
    def test_no_d_no_adv_before_start(self):
        t = _mk(gan=True, rgb=True, gan_start_step=10, gan_warmup_steps=10)
        d_steps = _count(t.discriminator_optimizer, "step")
        d_fwd = _count(t.discriminator, "forward")
        r = t.train_step(_imgs(), global_step=5)
        assert r["generator"] == 0.0 and r["weight_gan"] == 0.0
        assert not d_steps and not d_fwd

    @pytest.mark.parametrize("step,frac", [(10, 0.0), (15, 0.5), (20, 1.0), (500, 1.0)])
    def test_ramp(self, step, frac):
        t = _mk(
            gan=True,
            adaptive_weights=False,
            lambda_adv=0.5,
            gan_start_step=10,
            gan_warmup_steps=10,
        )
        assert t._train_generator(_imgs(), step)["weight_gan"] == pytest.approx(0.5 * frac)

    def test_d_trains_after_start_and_generator_returned(self):
        t = _mk(gan=True, gan_start_step=3, gan_warmup_steps=0)
        d_steps = _count(t.discriminator_optimizer, "step")
        r = t.train_step(_imgs(), global_step=3)
        assert len(d_steps) == 1 and "generator" in r and "discriminator" in r


class TestSchedulers:
    def test_once_per_distinct_global_step(self):
        t = _mk(gan=True)
        v = _count(t.scheduler, "step")
        d = _count(t.discriminator_scheduler, "step")
        for _ in range(3):
            t.train_step(_imgs(), global_step=5)
        t.train_step(_imgs(), global_step=6)
        assert len(v) == 2 and len(d) == 2

    def test_none_scheduler_ok(self):
        t = _mk(scheduler=None)
        t.train_step(_imgs(), global_step=3)

    def test_plateau_metric_is_plain_l1_without_hf_ramp(self):
        seen = []

        class _W:
            scheduler = ReduceLROnPlateau(torch.optim.SGD([nn.Parameter(torch.ones(1))], 0.1))

            def step(self, metric=None):
                seen.append(metric)

        t = _mk(scheduler=_W())
        imgs = _imgs()
        t.optimizer.param_groups[0]["lr"] = 0.0  # keep weights fixed between calls
        t.train_step(imgs, global_step=1)
        t.train_step(imgs, global_step=60000)
        assert seen[0] == pytest.approx(seen[1], rel=1e-4)


class TestDiscriminatorAccumulation:
    def test_freq2_accum2_d_steps_on_its_own_boundary(self):
        t = _mk(gan=True, discriminator_update_freq=2, gradient_accumulation_steps=2)
        d_step = _count(t.discriminator_optimizer, "step")
        d_zero = _count(t.discriminator_optimizer, "zero_grad")
        g_step = _count(t.optimizer, "step")
        for i in range(8):
            t.train_step(_imgs(), global_step=i)
        assert len(g_step) == 4
        assert len(d_step) == 2 and len(d_zero) == 2  # D ran on steps 0,2,4,6

    def test_g_nan_skip_keeps_counters_and_skips_d(self):
        t = _mk(gan=True, gradient_accumulation_steps=2)
        d_fwd = _count(t.discriminator, "forward")
        g_step = _count(t.optimizer, "step")
        bad = _imgs()
        bad[0, 0, 0, 0] = float("nan")
        t.train_step(_imgs(), 0)
        n_fwd = len(d_fwd)
        t.train_step(bad, 1)
        assert len(d_fwd) == n_fwd and t._accumulation_step == 1
        t.train_step(_imgs(), 2)
        assert len(g_step) == 1 and t._accumulation_step == 0


class TestR1Interval:
    def test_lazy_r1_scaled_by_interval(self):
        def loss(gamma, interval):
            torch.manual_seed(0)
            t = _mk(gan=True, r1_gamma=gamma, r1_interval=interval)
            torch.manual_seed(1)
            return t.train_step(_imgs(), global_step=0)["discriminator"]

        base, i1, i4 = loss(0.0, 1), loss(5.0, 1), loss(5.0, 4)
        assert i1 > base
        assert (i4 - base) == pytest.approx(4 * (i1 - base), rel=1e-3)


class TestReorderedDiscriminator:
    def test_single_compressor_and_expander_forward_per_step(self):
        t = _mk(gan=True)
        c = _count(t.compressor, "forward")
        e = _count(t.expander, "forward")
        t.train_step(_imgs(), global_step=1)
        assert len(c) == 1 and len(e) == 1

    def test_d_trains_on_detached_generator_output(self):
        t2 = _mk(gan=True)
        imgs = _imgs()
        t2._train_generator(imgs, 1)
        fake = t2.expander(t2.compressor(imgs, training=True)[0]).detach()
        t2.expander.scale.grad = None
        t2._train_discriminator(imgs, 1, fake)
        assert t2.expander.scale.grad is None

    def test_ctx_dim_zero_discriminator_ok(self):
        t = _mk(gan=True, d_ctx_dim=0)
        r = t.train_step(_imgs(), global_step=1)
        assert "discriminator" in r

    def test_ctx_dim_mismatch_still_raises_for_conditioned_d(self):
        t = _mk(gan=True, d_ctx_dim=_TOKEN_DIM + 1)
        with pytest.raises(RuntimeError, match="ctx_dim"):
            t.train_step(_imgs(), global_step=1)


class _InfAccelerator(_PlainAccelerator):
    """Backward that poisons/blows up grads of everything after backward."""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def backward(self, loss):
        loss.backward()
        for g in (p.grad for grp in self.params for p in grp if p.grad is not None):
            g.mul_(self.factor)


class TestFp16AndGradSafety:
    def test_d_nonfinite_grads_skip_step(self):
        t = _mk(gan=True)
        acc = _InfAccelerator(float("inf"))
        acc.params = [t.discriminator.parameters()]
        t.accelerator = acc
        d_step = _count(t.discriminator_optimizer, "step")
        r = t._train_discriminator(_imgs(), 1, _imgs())
        assert not d_step and r["_optimizer_stepped"] is False

    def test_g_nonfinite_grads_skip_step(self):
        t = _mk()
        acc = _InfAccelerator(float("inf"))
        acc.params = [t.compressor.parameters(), t.expander.parameters()]
        t.accelerator = acc
        g_step = _count(t.optimizer, "step")
        r = t._train_generator(_imgs(), 1)
        assert not g_step and r["_optimizer_stepped"] is False

    def test_d_grads_clipped(self):
        t = _mk(gan=True, gradient_clip_norm=1.0)
        acc = _InfAccelerator(1e6)
        acc.params = [t.discriminator.parameters()]
        t.accelerator = acc
        t.discriminator_optimizer.step = lambda: None
        t._train_discriminator(_imgs(), 1, _imgs())
        norm = torch.norm(
            torch.stack([p.grad.norm() for p in t.discriminator.parameters() if p.grad is not None])
        )
        assert norm <= 1.0 + 1e-4

    def test_scaler_updated_once_per_train_step_with_gan(self):
        acc = _FakeAccelerator()
        t = _mk(gan=True, accel=acc)
        calls = _count(acc.scaler, "update")
        t.train_step(_imgs(), global_step=1)
        assert len(calls) == 1

    def test_scaler_not_updated_when_nothing_unscaled(self):
        acc = _FakeAccelerator()
        t = _mk(gan=True, accel=acc, gradient_accumulation_steps=2)
        calls = _count(acc.scaler, "update")
        t.train_step(_imgs(), global_step=1)
        assert not calls
        t.train_step(_imgs(), global_step=2)
        assert len(calls) == 1


class TestCtxAuxDefault:
    def test_ctx_aux_off_by_default(self):
        assert _mk().train_ctx_aux is False
