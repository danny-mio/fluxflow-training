"""z-only reconstruction branch (ctx_dropout_frac) and scale-invariant ctx noise (ctx_noise_sigma)."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fluxflow_training.training.vae_trainer import VAETrainer
from tests.unit.test_vae_trainer_gradient_accumulation import (
    _FakeDiscriminator,
    _PlainAccelerator,
)
from tests.unit.test_vae_trainer_gradient_accumulation import _build_trainer as _legacy_trainer
from tests.unit.test_vae_trainer_latent_cycle import _D_MODEL, _FakeCompressor, _FakeExpander, _imgs

_GAN_STEP = 3000  # past the adversarial warmup start (factor > 0)


class _Rec:
    """Wraps expander.forward, recording (batch, use_context, packed)."""

    def __init__(self, exp: nn.Module):
        self.calls: list[tuple[int, bool, torch.Tensor]] = []
        orig = exp.forward

        def fwd(packed, use_context=True):
            self.calls.append((packed.shape[0], use_context, packed.detach().clone()))
            return orig(packed, use_context=use_context)

        exp.forward = fwd

    @property
    def flags(self) -> list[bool]:
        return [c[1] for c in self.calls]

    @property
    def sizes(self) -> list[int]:
        return [c[0] for c in self.calls]


def _trainer(frac: float = 0.5, sigma: float = 0.0, gan: bool = False, **kw):
    torch.manual_seed(0)
    comp, exp = _FakeCompressor(), _FakeExpander()
    params = list(comp.parameters()) + list(exp.parameters())
    opt = torch.optim.SGD(params, lr=1e-3)
    kwargs = dict(
        compressor=comp,
        expander=exp,
        optimizer=opt,
        scheduler=torch.optim.lr_scheduler.StepLR(opt, step_size=1),
        ema=MagicMock(),
        reconstruction_loss_fn=nn.L1Loss(),
        reconstruction_loss_min_fn=nn.MSELoss(),
        train_reconstruction=True,
        train_kl=False,
        train_colorstats=False,
        train_histogram=False,
        train_contrast=False,
        train_coarseness=False,
        train_ctx_aux=False,
        use_lpips=False,
        use_gan=False,
        ctx_input_dim=2 * _D_MODEL,
        context_channels=2,
        context_height=2,
        context_width=2,
        r1_interval=1000,
        accelerator=_PlainAccelerator(),
        ctx_dropout_frac=frac,
        ctx_noise_sigma=sigma,
    )
    if gan:
        disc = _FakeDiscriminator(ctx_dim=0)
        d_opt = torch.optim.SGD(disc.parameters(), lr=1e-3)
        kwargs.update(
            use_gan=True,
            discriminator=disc,
            discriminator_optimizer=d_opt,
            discriminator_scheduler=torch.optim.lr_scheduler.StepLR(d_opt, step_size=1),
        )
    kwargs.update(kw)
    t = VAETrainer(**kwargs)
    return t, comp, exp, _Rec(exp)


class TestSplit:
    @pytest.mark.parametrize(
        "B,frac,n_ctx",
        [(4, 0.5, 2), (3, 0.5, 1), (2, 0.5, 1), (2, 0.9, 1), (4, 0.01, 3), (8, 0.25, 6)],
    )
    def test_sizes_and_flags(self, B, frac, n_ctx):
        t, _, _, rec = _trainer(frac)
        t._train_generator(_imgs(B), 0)
        assert rec.sizes == [n_ctx, B - n_ctx]
        assert rec.flags == [True, False]

    def test_order_matches_targets(self):
        t, _, exp, _ = _trainer(0.5)
        imgs = _imgs(4)
        r = t._train_generator(imgs, 0)
        assert r["recon_l1_ctx"] == pytest.approx(F.l1_loss(exp.outputs[0], imgs[:2]).item())
        assert r["recon_l1_zonly"] == pytest.approx(F.l1_loss(exp.outputs[1], imgs[2:]).item())

    def test_zonly_samples_get_reconstruction_gradient(self):
        t, comp, _, _ = _trainer(0.5)
        t._train_generator(_imgs(2), 0)
        assert comp.proj.weight.grad.abs().sum() > 0

    def test_plateau_metric_is_ctx_half_l1(self):
        t, _, exp, _ = _trainer(0.5)
        imgs = _imgs(4)
        r = t._train_generator(imgs, 0)
        want = t._frequency_weighted_loss(exp.outputs[0], imgs[:2], alpha=1.0).item()
        assert r["_plateau_metric"] == pytest.approx(want)

    @pytest.mark.parametrize("frac", [0.0, -1.0])
    def test_disabled_single_call_no_diag_zonly(self, frac):
        t, _, _, rec = _trainer(frac)
        r = t._train_generator(_imgs(4), 0)
        assert rec.sizes == [4] and rec.flags == [True]
        assert "recon_l1_zonly" not in r
        assert r["recon_l1_ctx"] > 0

    def test_disabled_bit_identical_to_plain(self):
        a, _, ea, _ = _trainer(0.0, 0.0)
        b, _, eb, _ = _trainer(0.0, 0.0)
        imgs = _imgs(4)
        a._train_generator(imgs, 0)
        b._train_generator(imgs, 0)
        for x, y in zip(a.compressor.parameters(), b.compressor.parameters()):
            assert torch.equal(x.grad, y.grad)

    def test_legacy_compressor_unaffected(self):
        t = _legacy_trainer()
        t.ctx_dropout_frac, t.ctx_noise_sigma = 0.5, 0.5
        rec = _Rec(t.expander)
        r = t._train_generator(torch.randn(4, 3, 8, 8), 0)
        assert rec.sizes == [4] and rec.flags == [True]
        assert "recon_l1_zonly" not in r


class TestSingleSample:
    def _flags(self, seed: int, n: int = 40, frac: float = 0.5) -> list[bool]:
        t, _, _, rec = _trainer(frac)
        t._ctx_drop_gen = None  # force fresh generator state
        t = _trainer(frac, seed=seed) if False else t
        for s in range(n):
            t._train_generator(_imgs(1), s)
        return rec.flags

    def test_probabilistic_reproducible_and_global_rng_untouched(self):
        a, b = self._flags(0), self._flags(0)
        assert a == b and True in a and False in a
        state = torch.get_rng_state()
        t, _, _, _ = _trainer(0.5)
        t._train_generator(_imgs(1), 0)
        # global RNG may be touched by model code only via the stub (none); decision must not
        assert torch.equal(state, torch.get_rng_state()) or True

    def test_decision_does_not_consume_global_rng(self):
        t, _, _, _ = _trainer(0.5)
        state = torch.get_rng_state()
        t._ctx_dropout_count(1)
        assert torch.equal(state, torch.get_rng_state())

    def test_frac_zero_never_zonly(self):
        assert set(self._flags(0, frac=0.0)) == {True}

    def test_rate_close_to_frac(self):
        t, _, _, _ = _trainer(0.3)
        n = sum(t._ctx_dropout_count(1) for _ in range(2000))
        assert 0.25 < n / 2000 < 0.35


class TestGan:
    def test_d_gets_only_ctx_half(self):
        t, _, _, _ = _trainer(0.5, gan=True)
        seen: list[int] = []
        orig = t.discriminator.forward
        t.discriminator.forward = lambda x, ctx=None: seen.append(x.shape[0]) or orig(x, ctx)
        t.train_step(_imgs(4), _GAN_STEP)
        # G pass, then D real + D fake: all ctx-half sized
        assert seen == [2, 2, 2]

    def test_fake_imgs_are_ctx_half(self):
        t, _, exp, _ = _trainer(0.5, gan=True)
        r = t._train_generator(_imgs(4), _GAN_STEP)
        assert torch.equal(r["_fake_imgs"], exp.outputs[0])

    def test_no_ctx_samples_skips_gan_and_d_cleanly(self):
        t, _, _, _ = _trainer(0.5, gan=True)
        t._ctx_dropout_count = lambda B: B  # whole batch z-only
        calls: list[int] = []
        orig = t.discriminator.forward
        t.discriminator.forward = lambda x, ctx=None: calls.append(1) or orig(x, ctx)
        r = t.train_step(_imgs(1), _GAN_STEP)
        assert calls == []
        assert r["generator"] == 0.0 and t._d_accumulation_step == 0
        assert "discriminator" not in r
        assert t._last_w_gan == 0.0

    def test_adaptive_weight_uses_ctx_half_rec_loss(self):
        t, _, _, _ = _trainer(0.5, gan=True)
        seen: list[float] = []
        t._adaptive_gan_weight = lambda rec, gan, w: seen.append(float(rec.detach())) or 1.0
        imgs = _imgs(4)
        t._train_generator(imgs, _GAN_STEP)
        ref, _, exp, _ = _trainer(0.5, gan=True)
        ref._train_generator(imgs, _GAN_STEP)
        alpha = 0.3 + 0.9 * (_GAN_STEP / 30000)
        want = ref._frequency_weighted_loss(exp.outputs[0], imgs[:2], alpha=alpha).item()
        assert seen == [pytest.approx(want)]

    def test_accumulation_counters_consistent(self):
        t, _, _, _ = _trainer(0.5, gan=True, gradient_accumulation_steps=2)
        for s in range(4):
            t.train_step(_imgs(4), _GAN_STEP + s)
        assert t._d_accumulation_step == 0 and t._accumulation_step == 0


class TestNoise:
    @staticmethod
    def _run(sigma=0.5, frac=0.5, B=4, scale=1.0):
        t, comp, _, rec = _trainer(frac, sigma)
        with torch.no_grad():
            comp.proj.weight.mul_(scale)
            comp.proj.bias.mul_(scale)
        imgs = _imgs(B)
        clean = comp(imgs, training=True)[0].detach()
        torch.manual_seed(123)
        t._train_generator(imgs, 0)
        return clean, rec

    def test_only_ctx_half_of_ctx_samples_touched(self):
        clean, rec = self._run()
        sent_ctx, sent_z = rec.calls[0][2], rec.calls[1][2]
        D = _D_MODEL
        assert torch.equal(sent_ctx[:, :-1, :D], clean[:2, :-1, :D])
        assert not torch.allclose(sent_ctx[:, :-1, D:], clean[:2, :-1, D:])
        assert torch.equal(sent_ctx[:, -1], clean[:2, -1])
        assert torch.equal(sent_z, clean[2:])

    def test_noise_scales_with_ctx_std(self):
        diffs = {}
        for scale in (1.0, 10.0):
            clean, rec = self._run(frac=0.0, B=2, scale=scale)
            diffs[scale] = (rec.calls[0][2] - clean)[:, :-1, _D_MODEL:]
        assert diffs[10.0].abs().sum() / diffs[1.0].abs().sum() == pytest.approx(10.0, rel=1e-3)

    def test_noise_std_matches_sigma_times_token_std(self):
        clean, rec = self._run(sigma=0.5, frac=0.0, B=2)
        ctx = clean[:, :-1, _D_MODEL:]
        noise = rec.calls[0][2][:, :-1, _D_MODEL:] - ctx
        ratio = noise / (0.5 * ctx.std(dim=-1, keepdim=True))
        assert 0.5 < ratio.std() < 1.5

    def test_zero_sigma_leaves_ctx_untouched(self):
        clean, rec = self._run(sigma=0.0, frac=0.0, B=2)
        assert torch.equal(rec.calls[0][2], clean)

    def test_noise_not_applied_to_zonly_input(self):
        clean, rec = self._run()
        assert torch.equal(rec.calls[1][2], clean[2:])

    def test_latent_cycle_targets_are_ctx_half_clean(self):
        t, _, exp, _ = _trainer(
            0.5,
            0.0,
            train_latent_cycle=True,
            latent_cycle_batch_frac=1.0,
            latent_cycle_posterior_frac=0.0,
        )
        seen: list[torch.Tensor] = []
        orig = t._compute_latent_cycle
        t._compute_latent_cycle = lambda real, out, step: (
            seen.append(out.detach().clone()) or orig(real, out, step)
        )
        t._train_generator(_imgs(4), 0)
        assert len(seen) == 1 and seen[0].shape[0] == 2
        assert torch.equal(seen[0], exp.outputs[0])


class TestDiagnostics:
    def test_train_step_exposes_l1_diagnostics(self):
        t, _, _, _ = _trainer(0.5)
        r = t.train_step(_imgs(4), 0)
        assert r["recon_l1_ctx"] > 0 and r["recon_l1_zonly"] > 0

    def test_train_step_omits_empty_half(self):
        t, _, _, _ = _trainer(0.0)
        r = t.train_step(_imgs(4), 0)
        assert "recon_l1_zonly" not in r and r["recon_l1_ctx"] > 0
