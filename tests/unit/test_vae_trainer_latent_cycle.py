"""Unit tests for the v0.10.0 latent cycle loss (image-space G.E idempotence).

The compressor must re-encode an expander output into a latent that decodes to
the same image. The expander is frozen for this loss; only the compressor gets
gradient. Tiny stub compressor/expander, exercising ``_train_generator``
directly (see ``test_vae_trainer_bezier_reg.py`` for the pattern).
"""

import contextlib
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import fluxflow.models.v100.vae as vae_mod
from fluxflow.models.v100.vae import _ProgressiveUpscaler
from fluxflow_training.training.vae_trainer import VAETrainer

_D_MODEL = 4
_MAX_HW = 64


class _FakeBezier(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.p0 = nn.Parameter(torch.zeros(d_model))
        self.p1 = nn.Parameter(torch.full((d_model,), 0.33))
        self.p2 = nn.Parameter(torch.full((d_model,), 0.66))
        self.p3 = nn.Parameter(torch.ones(d_model))


class _FakeCompressor(nn.Module):
    """v0.10.0-shaped compressor stub; records every forward input."""

    def __init__(self, downscales: int = 1):
        super().__init__()
        self.d_model = _D_MODEL
        self.downscales = downscales
        self.max_hw = _MAX_HW
        self.use_gradient_checkpointing = False
        self.proj = nn.Conv2d(3, 2 * _D_MODEL, kernel_size=1)
        self.ctx_zinject_norm = nn.GroupNorm(2, _D_MODEL, affine=False)
        self.mu_activation = _FakeBezier(_D_MODEL)
        self.logvar_activation = _FakeBezier(_D_MODEL)
        self.inputs: list[torch.Tensor] = []

    @property
    def forward_calls(self) -> int:
        return len(self.inputs)

    def get_context_dims(self) -> int:
        return _D_MODEL

    def forward(self, x: torch.Tensor, training: bool = False):
        self.inputs.append(x.detach().clone())
        B, _, H, W = x.shape
        s = 2**self.downscales
        H_lat, W_lat = max(H // s, 1), max(W // s, 1)
        feat = F.adaptive_avg_pool2d(self.proj(x), (H_lat, W_lat))
        self.ctx_zinject_norm(feat[:, :_D_MODEL].contiguous())
        img_seq = feat.flatten(2).transpose(1, 2)
        hw_row = torch.zeros(B, 1, 2 * _D_MODEL, device=x.device, dtype=x.dtype)
        hw_row[:, 0, 0] = H_lat / float(_MAX_HW)
        hw_row[:, 0, 1] = W_lat / float(_MAX_HW)
        packed = torch.cat([img_seq, hw_row], dim=1)
        mu = torch.zeros(B, _D_MODEL, H_lat, W_lat, device=x.device, dtype=x.dtype)
        return packed, mu, torch.zeros_like(mu)


class _FakeExpander(nn.Module):
    """Expander stub recording outputs and param requires_grad at call time.

    ``sample_scale`` multiplies per-sample output, only for calls whose batch
    size equals its length (i.e. the main decode).
    """

    def __init__(self, downscales: int = 1):
        super().__init__()
        self.downscales = downscales
        self.max_hw = _MAX_HW
        self.proj = nn.Conv2d(2 * _D_MODEL, 3, kernel_size=1)
        self.outputs: list[torch.Tensor] = []
        self.flags_at_call: list[list[bool]] = []
        self.sample_scale: torch.Tensor | None = None
        self.nan_from_call: int | None = None
        self.raise_from_call: int | None = None

    @property
    def forward_calls(self) -> int:
        return len(self.outputs)

    def forward(self, packed: torch.Tensor, use_context: bool = True) -> torch.Tensor:
        n = len(self.outputs) + 1
        self.flags_at_call.append([self.proj.weight.requires_grad, self.proj.bias.requires_grad])
        if self.raise_from_call is not None and n >= self.raise_from_call:
            self.outputs.append(packed)
            raise RuntimeError("expander boom")
        B, _, D2 = packed.shape
        hw = packed[:, -1, :]
        H_lat = max(int(round(hw[0, 0].item() * self.max_hw)), 1)
        W_lat = max(int(round(hw[0, 1].item() * self.max_hw)), 1)
        feat = packed[:, :-1, :].transpose(1, 2).reshape(B, D2, H_lat, W_lat)
        out = F.interpolate(self.proj(feat), scale_factor=2**self.downscales, mode="nearest")
        if self.sample_scale is not None and len(self.sample_scale) == B:
            out = out * self.sample_scale.view(B, 1, 1, 1)
        if self.nan_from_call is not None and n >= self.nan_from_call:
            out = out * float("nan")
        self.outputs.append(out.detach().clone())
        return out


class _PlainAccelerator:
    def __init__(self):
        self.scaler = None

    def backward(self, loss):
        loss.backward()

    def unscale_gradients(self, optimizer=None):
        pass

    def autocast(self):
        return contextlib.nullcontext()


class _FakeLPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[tuple[torch.dtype, torch.dtype]] = []

    def forward(self, a, b):
        self.calls.append((a.dtype, b.dtype))
        return (a - b).abs().mean(dim=(1, 2, 3), keepdim=True)


def _build(
    downscales: int = 1,
    seed: int = 0,
    train_reconstruction: bool = False,
    ctx_shrinkage_weight: float = 0.0,
    accum: int = 1,
    **cycle,
):
    cycle.setdefault("ctx_dropout_frac", 0.0)  # z-only split has its own tests
    cycle.setdefault("ctx_noise_sigma", 0.0)
    torch.manual_seed(seed)
    compressor = _FakeCompressor(downscales)
    expander = _FakeExpander(downscales)
    opt = torch.optim.SGD(list(compressor.parameters()) + list(expander.parameters()), lr=1e-3)
    trainer = VAETrainer(
        compressor=compressor,
        expander=expander,
        optimizer=opt,
        scheduler=torch.optim.lr_scheduler.StepLR(opt, step_size=1),
        ema=MagicMock(),
        reconstruction_loss_fn=nn.L1Loss(),
        reconstruction_loss_min_fn=nn.MSELoss(),
        train_reconstruction=train_reconstruction,
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
        gradient_accumulation_steps=accum,
        ctx_shrinkage_weight=ctx_shrinkage_weight,
        ctx_shrinkage_warmup_start_step=0,
        ctx_shrinkage_warmup_steps=0,
        **cycle,
    )
    return trainer, compressor, expander


def _imgs(B: int = 2, H: int = 8, W: int = 8, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, 3, H, W, generator=g) * 2 - 1


_ON = {"train_latent_cycle": True}


class TestNotComputed:
    @pytest.mark.parametrize(
        "kwargs,step",
        [
            ({}, 0),  # disabled
            ({"train_latent_cycle": True, "latent_cycle_start_step": 10}, 5),
            ({"train_latent_cycle": True, "latent_cycle_every": 4}, 3),
            (
                {
                    "train_latent_cycle": True,
                    "latent_cycle_start_step": 5,
                    "latent_cycle_warmup_steps": 10,
                },
                5,
            ),  # lambda still 0 at start
            ({"train_latent_cycle": True, "lambda_latent_cycle": 0.0}, 0),
        ],
    )
    def test_no_extra_forwards(self, kwargs, step):
        trainer, comp, exp = _build(**kwargs)
        result = trainer._train_generator(_imgs(), global_step=step)
        assert comp.forward_calls == 1
        assert exp.forward_calls == 1
        assert result["latent_cycle_loss"] == 0.0
        assert result["latent_cycle_reject_frac"] == 0.0

    def test_computed_on_cadence_and_after_start(self):
        trainer, comp, exp = _build(latent_cycle_every=4, latent_cycle_start_step=4, **_ON)
        result = trainer._train_generator(_imgs(), global_step=8)
        assert comp.forward_calls == 2
        assert exp.forward_calls == 2
        assert result["latent_cycle_loss"] > 0.0

    def test_keys_present_in_mid_accumulation_return(self):
        trainer, _, _ = _build(accum=2, **_ON)
        result = trainer._train_generator(_imgs(), global_step=0)
        assert result["_optimizer_stepped"] is False
        assert result["latent_cycle_loss"] > 0.0
        assert "latent_cycle_reject_frac" in result

    def test_keys_present_in_nan_input_return(self):
        trainer, _, _ = _build(**_ON)
        bad = _imgs()
        bad[0, 0, 0, 0] = float("nan")
        result = trainer._train_generator(bad, global_step=0)
        assert result["latent_cycle_loss"] == 0.0
        assert result["latent_cycle_reject_frac"] == 0.0


class TestLambdaSchedule:
    def test_ramp(self):
        trainer, _, _ = _build(
            lambda_latent_cycle=2.0,
            latent_cycle_start_step=100,
            latent_cycle_warmup_steps=50,
            **_ON,
        )
        lam = trainer._latent_cycle_lambda
        assert lam(0) == 0.0
        assert lam(99) == 0.0
        assert lam(100) == 0.0
        assert lam(125) == pytest.approx(1.0)
        assert lam(150) == pytest.approx(2.0)
        assert lam(10_000) == pytest.approx(2.0)

    def test_no_warmup_is_step_function(self):
        trainer, _, _ = _build(lambda_latent_cycle=3.0, latent_cycle_start_step=10, **_ON)
        assert trainer._latent_cycle_lambda(9) == 0.0
        assert trainer._latent_cycle_lambda(10) == 3.0


class TestGradients:
    @pytest.mark.parametrize("posterior_frac", [0.0, 1.0])
    def test_cycle_only_expander_gets_no_grad_compressor_does(self, posterior_frac):
        trainer, comp, exp = _build(latent_cycle_posterior_frac=posterior_frac, **_ON)
        trainer._train_generator(_imgs(), global_step=0)
        for p in exp.parameters():
            assert p.grad is None or torch.all(p.grad == 0)
        assert comp.proj.weight.grad is not None
        assert comp.proj.weight.grad.abs().sum().item() > 0

    @pytest.mark.parametrize("posterior_frac", [0.0, 0.5, 1.0])
    def test_expander_grads_bitwise_identical_with_cycle_on_vs_off(self, posterior_frac):
        t_off, _, e_off = _build(train_reconstruction=True)
        t_on, _, e_on = _build(
            train_reconstruction=True, latent_cycle_posterior_frac=posterior_frac, **_ON
        )
        t_off._train_generator(_imgs(4), global_step=0)
        t_on._train_generator(_imgs(4), global_step=0)
        grads_off = [p.grad for p in e_off.parameters()]
        grads_on = [p.grad for p in e_on.parameters()]
        assert all(g is not None for g in grads_off)
        for a, b in zip(grads_off, grads_on):
            assert torch.equal(a, b)

    def test_lambda_scales_gradient_not_reported_loss(self):
        t1, c1, _ = _build(lambda_latent_cycle=1.0, **_ON)
        t3, c3, _ = _build(lambda_latent_cycle=3.0, **_ON)
        r1 = t1._train_generator(_imgs(), global_step=0)
        r3 = t3._train_generator(_imgs(), global_step=0)
        assert r1["latent_cycle_loss"] == pytest.approx(r3["latent_cycle_loss"])
        g1, g3 = c1.proj.weight.grad, c3.proj.weight.grad
        # bezier reg / context terms carry no grad on proj, so the ratio is exact.
        assert torch.allclose(g3, 3.0 * g1, rtol=1e-4, atol=1e-8)


class _CkptExpander(_FakeExpander):
    """Expander routing features through a real checkpointed upscaler."""

    def __init__(self, downscales: int = 1):
        super().__init__(downscales)
        self.up = _ProgressiveUpscaler(
            channels=2 * _D_MODEL, steps=1, use_spade=False, use_gradient_checkpointing=True
        )
        self.head = nn.Conv2d(self.up.out_channels, 3, kernel_size=1)

    def forward(self, packed: torch.Tensor, use_context: bool = True) -> torch.Tensor:
        B, _, D2 = packed.shape
        hw = packed[:, -1, :]
        H_lat = max(int(round(hw[0, 0].item() * self.max_hw)), 1)
        W_lat = max(int(round(hw[0, 1].item() * self.max_hw)), 1)
        feat = packed[:, :-1, :].transpose(1, 2).reshape(B, D2, H_lat, W_lat)
        out = self.head(self.up(feat, None))  # upscaler supplies the 2x of downscales=1
        self.outputs.append(out.detach().clone())
        return out


class TestCheckpointedFrozenExpander:
    def _trainer(self):
        trainer, comp, _ = _build(**_ON)
        trainer.expander = _CkptExpander()
        return trainer, comp, trainer.expander

    def test_cycle_grads_only_reach_compressor(self):
        trainer, comp, exp = self._trainer()
        loss, _ = trainer._compute_latent_cycle(_imgs(), _imgs(seed=2), 0)
        loss.backward()
        assert all(p.grad is None for p in exp.parameters())
        assert comp.proj.weight.grad.abs().sum().item() > 0

    def test_checkpointing_used_for_cycle_decode(self, monkeypatch):
        trainer, _, exp = self._trainer()
        calls: list[int] = []
        real = vae_mod.checkpoint
        monkeypatch.setattr(vae_mod, "checkpoint", lambda *a, **k: calls.append(1) or real(*a, **k))
        trainer._compute_latent_cycle(_imgs(), _imgs(seed=2), 0)[0].backward()
        assert len(calls) == len(exp.up.layers)

    def test_flags_never_mutated(self):
        trainer, _, exp = self._trainer()
        trainer._train_generator(_imgs(), global_step=0)
        assert all(p.requires_grad for p in exp.parameters())


class TestExpanderFreezing:
    def test_expander_frozen_during_cycle_decode_only(self):
        trainer, _, exp = _build(**_ON)
        trainer._train_generator(_imgs(), global_step=0)
        assert exp.flags_at_call[0] == [True, True]  # main decode
        assert exp.flags_at_call[1] == [False, False]  # cycle decode

    def test_flags_restored_after_step(self):
        trainer, _, exp = _build(**_ON)
        exp.proj.bias.requires_grad_(False)
        trainer._train_generator(_imgs(), global_step=0)
        assert exp.proj.weight.requires_grad is True
        assert exp.proj.bias.requires_grad is False

    def test_flags_restored_when_expander_raises_in_cycle(self):
        trainer, _, exp = _build(**_ON)
        exp.raise_from_call = 2
        exp.proj.bias.requires_grad_(False)
        result = trainer._train_generator(_imgs(), global_step=0)
        assert exp.proj.weight.requires_grad is True
        assert exp.proj.bias.requires_grad is False
        assert result["latent_cycle_loss"] == 0.0  # non-fatal


class TestSubBatchAndCrop:
    def test_batch_frac_uses_first_ceil_samples(self):
        trainer, comp, _ = _build(latent_cycle_batch_frac=0.5, **_ON)
        trainer._train_generator(_imgs(3), global_step=0)
        assert comp.inputs[1].shape[0] == 2  # ceil(0.5 * 3)

    def test_batch_frac_minimum_one(self):
        trainer, comp, _ = _build(latent_cycle_batch_frac=0.01, **_ON)
        trainer._train_generator(_imgs(3), global_step=0)
        assert comp.inputs[1].shape[0] == 1

    @pytest.mark.parametrize("downscales,frac,expected", [(2, 0.6, 8), (2, 0.01, 4), (1, 1.0, 16)])
    def test_crop_window_size_rounded_down_aligned_min_cell(self, downscales, frac, expected):
        trainer, _, _ = _build(downscales=downscales, latent_cycle_crop_frac=frac, **_ON)
        unit = 2**downscales
        for _ in range(30):
            top, left, h, w = trainer._latent_cycle_crop_window(16, 16, unit)
            assert (h, w) == (expected, expected)
            assert top % unit == 0 and left % unit == 0
            assert 0 <= top <= 16 - h and 0 <= left <= 16 - w

    def test_crop_window_nonsquare_and_nonmultiple_dims(self):
        trainer, _, _ = _build(downscales=2, latent_cycle_crop_frac=0.5, **_ON)
        seen = set()
        for _ in range(100):
            top, left, h, w = trainer._latent_cycle_crop_window(18, 30, 4)
            assert (h, w) == (8, 12)  # floor(9/4)*4, floor(15/4)*4
            assert top % 4 == 0 and left % 4 == 0
            assert top + h <= 18 and left + w <= 30
            seen.add((top, left))
        assert len(seen) > 1  # offsets are random

    def test_cycle_compressor_input_has_cropped_shape(self):
        trainer, comp, _ = _build(downscales=2, latent_cycle_crop_frac=0.6, **_ON)
        trainer._train_generator(_imgs(2, 16, 16), global_step=0)
        assert comp.inputs[1].shape == (2, 3, 8, 8)

    def test_c1a_source_is_cropped_recon_with_fixed_window(self):
        trainer, comp, exp = _build(downscales=2, latent_cycle_crop_frac=0.5, **_ON)
        trainer._latent_cycle_crop_window = lambda H, W, unit: (4, 8, 8, 8)
        trainer._train_generator(_imgs(2, 16, 16), global_step=0)
        assert torch.equal(comp.inputs[1], exp.outputs[0][:, :, 4:12, 8:16])

    def test_c1b_source_crop_applied_to_real_images_before_encoding(self):
        trainer, comp, _ = _build(
            downscales=2, latent_cycle_crop_frac=0.5, latent_cycle_posterior_frac=1.0, **_ON
        )
        trainer._latent_cycle_crop_window = lambda H, W, unit: (4, 8, 8, 8)
        imgs = _imgs(2, 16, 16)
        trainer._train_generator(imgs, global_step=0)
        assert torch.equal(comp.inputs[1], imgs[:, :, 4:12, 8:16])


class TestPosteriorFrac:
    def test_zero_has_no_extra_compressor_forward_on_real_images(self):
        trainer, comp, exp = _build(latent_cycle_posterior_frac=0.0, **_ON)
        trainer._train_generator(_imgs(4), global_step=0)
        assert comp.forward_calls == 2  # main + p'
        assert exp.forward_calls == 2  # main + frozen decode

    def test_one_adds_exactly_one_forward(self):
        trainer, comp, exp = _build(latent_cycle_posterior_frac=1.0, **_ON)
        imgs = _imgs(4)
        trainer._train_generator(imgs, global_step=0)
        assert comp.forward_calls == 3  # main + posterior encode + p'
        assert torch.equal(comp.inputs[1], imgs)
        assert exp.forward_calls == 3  # main + C1b decode + frozen decode

    def test_mixed_split(self):
        trainer, comp, exp = _build(latent_cycle_posterior_frac=0.5, **_ON)
        imgs = _imgs(4)
        trainer._train_generator(imgs, global_step=0)
        assert comp.forward_calls == 3
        assert comp.inputs[1].shape[0] == 2  # posterior part
        assert comp.inputs[2].shape[0] == 4  # full cycle batch re-encoded
        # C1b part is last 2 samples; C1a part (first 2) is the main recon.
        assert torch.equal(comp.inputs[1], imgs[2:])
        assert torch.equal(comp.inputs[2][:2], exp.outputs[0][:2])


class TestRejectGuard:
    def _scaled(self, scales, **kw):
        trainer, comp, exp = _build(**{**_ON, **kw})
        exp.sample_scale = torch.tensor(scales)
        return trainer, comp, exp

    def test_out_of_range_sample_dropped_and_frac_reported(self):
        trainer, comp, _ = self._scaled([1.0, 1000.0, 1.0])
        result = trainer._train_generator(_imgs(3), global_step=0)
        assert comp.inputs[1].shape[0] == 2
        assert result["latent_cycle_reject_frac"] == pytest.approx(1 / 3)
        assert result["latent_cycle_loss"] > 0.0

    def test_all_rejected_gives_zero_loss_without_error(self):
        trainer, comp, exp = self._scaled([1000.0, 1000.0])
        result = trainer._train_generator(_imgs(2), global_step=0)
        assert result["latent_cycle_loss"] == 0.0
        assert result["latent_cycle_reject_frac"] == 1.0
        assert comp.forward_calls == 1
        assert exp.forward_calls == 1

    def test_no_clamp_applied_to_cycle_input(self):
        trainer, comp, _ = self._scaled([1000.0, 1.0], latent_cycle_reject_frac=1.0)
        trainer._train_generator(_imgs(2), global_step=0)
        assert comp.inputs[1].abs().max() > 1.5  # kept as-is

    def test_reject_range_and_frac_thresholds(self):
        trainer, _, _ = _build(latent_cycle_reject_range=1.0, latent_cycle_reject_frac=0.25, **_ON)
        x = torch.zeros(2, 3, 4, 4)
        x[0, 0, :2, :2] = 2.0  # 4 / 48 elems out of range -> kept
        x[1, :, :2, :] = 2.0  # 24 / 48 -> rejected
        keep = trainer._latent_cycle_keep_mask(x)
        assert keep.tolist() == [True, False]

    def test_c1b_samples_also_guarded(self):
        trainer, comp, exp = _build(latent_cycle_posterior_frac=1.0, **_ON)
        orig = exp.forward

        def big_first_call(packed, use_context=True):
            out = orig(packed, use_context)
            return out * 1000.0 if len(exp.outputs) == 2 else out

        exp.forward = big_first_call  # second call = C1b decode
        result = trainer._train_generator(_imgs(2), global_step=0)
        assert result["latent_cycle_reject_frac"] == 1.0
        assert result["latent_cycle_loss"] == 0.0


class TestNaNGuard:
    def test_nan_in_x_hat_zeroes_cycle_loss_and_step_continues(self, caplog):
        trainer, _, exp = _build(train_reconstruction=True, **_ON)
        exp.nan_from_call = 2
        with caplog.at_level("WARNING"):
            result = trainer._train_generator(_imgs(), global_step=0)
        assert result["latent_cycle_loss"] == 0.0
        assert result["recon"] > 0.0
        assert result["_optimizer_stepped"] is True
        assert any("latent_cycle" in r.message for r in caplog.records)


class TestLPIPSTerm:
    def test_lpips_adds_to_loss_in_fp32(self):
        t0, _, _ = _build(**_ON)
        t1, _, _ = _build(latent_cycle_lpips_weight=2.0, **_ON)
        t1.lpips_fn = _FakeLPIPS()
        r0 = t0._train_generator(_imgs(), global_step=0)
        r1 = t1._train_generator(_imgs(), global_step=0)
        assert len(t1.lpips_fn.calls) == 1
        assert t1.lpips_fn.calls[0] == (torch.float32, torch.float32)
        assert r1["latent_cycle_loss"] > r0["latent_cycle_loss"]

    def test_weight_without_lpips_fn_is_ignored(self):
        t0, _, _ = _build(**_ON)
        t1, _, _ = _build(latent_cycle_lpips_weight=2.0, **_ON)
        t1.lpips_fn = None
        r0 = t0._train_generator(_imgs(), global_step=0)
        r1 = t1._train_generator(_imgs(), global_step=0)
        assert r1["latent_cycle_loss"] == pytest.approx(r0["latent_cycle_loss"])

    def test_zero_weight_never_calls_lpips(self):
        t, _, _ = _build(**_ON)
        t.lpips_fn = _FakeLPIPS()
        t._train_generator(_imgs(), global_step=0)
        assert t.lpips_fn.calls == []


class TestCycleOutsideReconWeighting:
    def test_recon_term_unaffected_by_cycle(self):
        imgs = _imgs()
        t_off, _, _ = _build(train_reconstruction=True)
        t_on, _, _ = _build(train_reconstruction=True, **_ON)
        r_off = t_off._train_generator(imgs, global_step=0)
        r_on = t_on._train_generator(imgs, global_step=0)
        assert r_on["recon"] == pytest.approx(r_off["recon"])


class TestCtxShrinkageNotOverwrittenByCycle:
    @pytest.mark.parametrize("posterior_frac", [0.0, 1.0])
    def test_shrinkage_loss_identical_with_cycle_on_vs_off(self, posterior_frac):
        imgs = _imgs(2, seed=3)
        t_off, _, _ = _build(ctx_shrinkage_weight=0.5)
        t_on, _, _ = _build(
            ctx_shrinkage_weight=0.5, latent_cycle_posterior_frac=posterior_frac, **_ON
        )
        r_off = t_off._train_generator(imgs, global_step=0)
        r_on = t_on._train_generator(imgs, global_step=0)
        assert r_off["ctx_shrinkage_loss"] > 0.0
        assert r_on["ctx_shrinkage_loss"] == r_off["ctx_shrinkage_loss"]

    def test_capture_re_enabled_after_step(self):
        trainer, _, _ = _build(ctx_shrinkage_weight=0.5, **_ON)
        trainer._train_generator(_imgs(), global_step=0)
        assert trainer._ctx_capture_enabled is True
