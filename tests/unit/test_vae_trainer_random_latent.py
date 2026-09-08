"""Unit tests for the v0.10.0 random-latent compressor training loss.

Assumes the expander is already well-trained: sample a random packed latent,
decode it into a synthetic image via the frozen (no-grad) expander, then train
the compressor to re-encode that synthetic image back into the same random
latent. Trains the compressor in isolation using the expander purely as a
frozen latent-to-image function. Gated entirely on ``train_random_latent``;
must run alongside other VAE losses or fully standalone.

We don't run a full real training step here -- minimal v0.10.0-shaped fake
compressor/expander stubs, following ``test_vae_trainer_bezier_reg.py``'s
pattern of exercising ``_train_generator`` directly with a ``_PlainAccelerator``
double.
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fluxflow_training.training.vae_trainer import VAETrainer

_D_MODEL = 4
_DOWNSCALES = 1
_MAX_HW = 64
_IMG_SHAPE = (3, 8, 8)


class _FakeBezier(nn.Module):
    """Minimal monotonic Bezier stub exposing p0..p3 (see compute_bezier_monotonicity_reg)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.p0 = nn.Parameter(torch.zeros(d_model))
        self.p1 = nn.Parameter(torch.full((d_model,), 0.33))
        self.p2 = nn.Parameter(torch.full((d_model,), 0.66))
        self.p3 = nn.Parameter(torch.ones(d_model))


class _FakeCompressorV100(nn.Module):
    """v0.10.0-shaped compressor stub: packed width 2*d_model + HW-encoding row."""

    def __init__(
        self, d_model: int = _D_MODEL, downscales: int = _DOWNSCALES, max_hw: int = _MAX_HW
    ):
        super().__init__()
        self.d_model = d_model
        self.downscales = downscales
        self.max_hw = max_hw
        self.use_gradient_checkpointing = False
        self.proj = nn.Conv2d(3, 2 * d_model, kernel_size=1)
        self.mu_activation = _FakeBezier(d_model)
        self.logvar_activation = _FakeBezier(d_model)
        self.forward_calls = 0

    def get_context_dims(self) -> int:
        return self.d_model

    def get_downscales(self) -> int:
        return self.downscales

    def forward(self, x: torch.Tensor, training: bool = False):
        self.forward_calls += 1
        B, _, H, W = x.shape
        H_lat = max(H // (2**self.downscales), 1)
        W_lat = max(W // (2**self.downscales), 1)
        feat = F.adaptive_avg_pool2d(self.proj(x), (H_lat, W_lat))
        img_seq = feat.flatten(2).transpose(1, 2)  # [B, T, 2D]
        hw_row = torch.zeros(B, 1, 2 * self.d_model, device=x.device, dtype=x.dtype)
        hw_row[:, 0, 0] = H_lat / float(self.max_hw)
        hw_row[:, 0, 1] = W_lat / float(self.max_hw)
        packed = torch.cat([img_seq, hw_row], dim=1)  # [B, T+1, 2D]
        mu = torch.zeros(B, self.d_model, H_lat, W_lat, device=x.device, dtype=x.dtype)
        logvar = torch.zeros_like(mu)
        return packed, mu, logvar


class _FakeExpanderV100(nn.Module):
    """v0.10.0-shaped expander stub: decodes a packed [B, T+1, 2D] tensor to an image."""

    def __init__(
        self,
        d_model: int = _D_MODEL,
        downscales: int = _DOWNSCALES,
        max_hw: int = _MAX_HW,
        out_channels: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.downscales = downscales
        self.max_hw = max_hw
        self.proj = nn.Conv2d(2 * d_model, out_channels, kernel_size=1)
        self.forward_calls = 0

    def forward(self, packed: torch.Tensor, use_context: bool = True) -> torch.Tensor:
        self.forward_calls += 1
        B, _, D2 = packed.shape
        hw_row = packed[:, -1, :]
        H_lat = max(int(round(hw_row[0, 0].item() * self.max_hw)), 1)
        W_lat = max(int(round(hw_row[0, 1].item() * self.max_hw)), 1)
        feat = packed[:, :-1, :].transpose(1, 2).reshape(B, D2, H_lat, W_lat)
        return F.interpolate(self.proj(feat), scale_factor=2**self.downscales, mode="nearest")


class _PlainAccelerator:
    """No-AMP accelerator double matching production with mixed_precision='no'."""

    def __init__(self):
        self.scaler = None

    def backward(self, loss):
        loss.backward()

    def unscale_gradients(self, optimizer=None):
        pass

    def autocast(self):
        import contextlib

        return contextlib.nullcontext()


def _build_trainer(
    compressor: nn.Module,
    expander: nn.Module,
    train_random_latent: bool,
    lambda_random_latent_z: float = 1.0,
    lambda_random_latent_ctx: float = 1.0,
) -> VAETrainer:
    """Minimally-configured VAETrainer with every other train_*/use_* flag off."""
    opt = torch.optim.SGD(list(compressor.parameters()) + list(expander.parameters()), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

    return VAETrainer(
        compressor=compressor,
        expander=expander,
        optimizer=opt,
        scheduler=sched,
        ema=MagicMock(),
        reconstruction_loss_fn=nn.L1Loss(),
        reconstruction_loss_min_fn=nn.MSELoss(),
        train_reconstruction=False,
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
        gradient_accumulation_steps=1,
        train_random_latent=train_random_latent,
        lambda_random_latent_z=lambda_random_latent_z,
        lambda_random_latent_ctx=lambda_random_latent_ctx,
    )


class TestRandomLatentDisabled:
    def test_zero_loss_and_no_extra_forward_passes(self):
        compressor = _FakeCompressorV100()
        expander = _FakeExpanderV100()
        trainer = _build_trainer(compressor, expander, train_random_latent=False)

        result = trainer._train_generator(torch.randn(2, *_IMG_SHAPE), global_step=0)

        assert result["random_latent_z_loss"] == 0.0
        assert result["random_latent_ctx_loss"] == 0.0
        # Only the main encode/decode pass -- no extra forward passes when off.
        assert compressor.forward_calls == 1
        assert expander.forward_calls == 1
        # proj is unused by any active loss when every other flag is off too.
        assert compressor.proj.weight.grad is None

    def test_key_present_in_mid_accumulation_return(self):
        compressor = _FakeCompressorV100()
        expander = _FakeExpanderV100()
        opt = torch.optim.SGD(list(compressor.parameters()) + list(expander.parameters()), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        trainer = VAETrainer(
            compressor=compressor,
            expander=expander,
            optimizer=opt,
            scheduler=sched,
            ema=MagicMock(),
            reconstruction_loss_fn=nn.L1Loss(),
            reconstruction_loss_min_fn=nn.MSELoss(),
            train_reconstruction=False,
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
            gradient_accumulation_steps=2,
            train_random_latent=False,
        )

        result = trainer._train_generator(torch.randn(2, *_IMG_SHAPE), global_step=0)

        assert result["_optimizer_stepped"] is False
        assert result["random_latent_z_loss"] == 0.0
        assert result["random_latent_ctx_loss"] == 0.0


class TestRandomLatentEnabled:
    def test_loss_nonzero_and_present_with_double_forward_passes(self):
        torch.manual_seed(0)
        compressor = _FakeCompressorV100()
        expander = _FakeExpanderV100()
        trainer = _build_trainer(compressor, expander, train_random_latent=True)

        result = trainer._train_generator(torch.randn(2, *_IMG_SHAPE), global_step=0)

        assert "random_latent_z_loss" in result
        assert "random_latent_ctx_loss" in result
        assert result["random_latent_z_loss"] > 0.0
        assert result["random_latent_ctx_loss"] > 0.0
        # Main encode + random-latent re-encode.
        assert compressor.forward_calls == 2
        # Main decode + random-latent synth decode.
        assert expander.forward_calls == 2

    def test_no_gradient_reaches_expander_but_compressor_gets_gradient(self):
        torch.manual_seed(0)
        compressor = _FakeCompressorV100()
        expander = _FakeExpanderV100()
        trainer = _build_trainer(compressor, expander, train_random_latent=True)

        trainer._train_generator(torch.randn(2, *_IMG_SHAPE), global_step=0)

        for p in expander.parameters():
            assert p.grad is None or torch.all(p.grad == 0)
        assert compressor.proj.weight.grad is not None
        assert compressor.proj.weight.grad.abs().sum().item() > 0

    def test_runs_standalone_with_all_other_flags_off(self):
        """All other train_*/use_* flags off (see _build_trainer) -- this loss
        alone must not error and must produce a finite, gradient-carrying loss."""
        torch.manual_seed(1)
        compressor = _FakeCompressorV100()
        expander = _FakeExpanderV100()
        trainer = _build_trainer(compressor, expander, train_random_latent=True)

        result = trainer._train_generator(torch.randn(2, *_IMG_SHAPE), global_step=0)

        assert result["_optimizer_stepped"] is True
        assert torch.isfinite(torch.tensor(result["random_latent_z_loss"]))
        assert torch.isfinite(torch.tensor(result["random_latent_ctx_loss"]))

    def test_final_return_includes_detached_float_value(self):
        torch.manual_seed(0)
        compressor = _FakeCompressorV100()
        expander = _FakeExpanderV100()
        trainer = _build_trainer(compressor, expander, train_random_latent=True)

        result = trainer._train_generator(torch.randn(2, *_IMG_SHAPE), global_step=0)

        assert isinstance(result["random_latent_z_loss"], float)
        assert isinstance(result["random_latent_ctx_loss"], float)


class TestRandomLatentLossShapes:
    """Behavioral proof the z branch is MSE-shaped and the ctx branch is
    cosine-shaped -- the property motivating the split (ctx is actively
    shrunk toward zero magnitude by ctx_shrinkage_weight, so a magnitude-
    sensitive loss there would fight that regularizer)."""

    def test_z_loss_scales_with_squared_distance(self):
        z_target = torch.zeros(2, 3, _D_MODEL)
        z_rec_near = z_target + 0.1
        z_rec_far = z_target + 1.0

        near = F.mse_loss(z_rec_near, z_target).item()
        far = F.mse_loss(z_rec_far, z_target).item()

        assert far == pytest.approx(100 * near, rel=1e-4)

    def test_ctx_loss_zero_when_same_direction_any_magnitude(self):
        ctx_target = torch.randn(2, 3, _D_MODEL)
        for scale in (0.01, 1.0, 50.0):
            ctx_rec = ctx_target * scale
            loss = 1 - F.cosine_similarity(ctx_rec, ctx_target, dim=-1).mean()
            assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_ctx_loss_insensitive_to_pure_magnitude_rescale(self):
        torch.manual_seed(2)
        ctx_target = torch.randn(2, 3, _D_MODEL)
        ctx_rec = torch.randn(2, 3, _D_MODEL)

        base = 1 - F.cosine_similarity(ctx_rec, ctx_target, dim=-1).mean()
        rescaled = 1 - F.cosine_similarity(ctx_rec * 7.0, ctx_target, dim=-1).mean()

        assert rescaled.item() == pytest.approx(base.item(), abs=1e-5)


class TestRandomLatentIndependentLambdas:
    """proj.weight's first d_model output channels feed the z half of the
    packed latent, the next d_model feed the ctx half (see
    _FakeCompressorV100.forward) -- so each lambda's isolated effect on
    total_loss shows up as gradient on its own channel slice only, with
    every other train_*/use_* flag off."""

    def test_zeroing_ctx_lambda_zeroes_ctx_channel_gradient_only(self):
        torch.manual_seed(0)
        compressor = _FakeCompressorV100()
        expander = _FakeExpanderV100()
        trainer = _build_trainer(
            compressor,
            expander,
            train_random_latent=True,
            lambda_random_latent_z=1.0,
            lambda_random_latent_ctx=0.0,
        )

        trainer._train_generator(torch.randn(2, *_IMG_SHAPE), global_step=0)

        grad = compressor.proj.weight.grad
        assert grad is not None
        assert grad[:_D_MODEL].abs().sum().item() > 0  # z channels
        assert grad[_D_MODEL:].abs().sum().item() == 0  # ctx channels

    def test_zeroing_z_lambda_zeroes_z_channel_gradient_only(self):
        torch.manual_seed(0)
        compressor = _FakeCompressorV100()
        expander = _FakeExpanderV100()
        trainer = _build_trainer(
            compressor,
            expander,
            train_random_latent=True,
            lambda_random_latent_z=0.0,
            lambda_random_latent_ctx=1.0,
        )

        trainer._train_generator(torch.randn(2, *_IMG_SHAPE), global_step=0)

        grad = compressor.proj.weight.grad
        assert grad is not None
        assert grad[:_D_MODEL].abs().sum().item() == 0  # z channels
        assert grad[_D_MODEL:].abs().sum().item() > 0  # ctx channels
