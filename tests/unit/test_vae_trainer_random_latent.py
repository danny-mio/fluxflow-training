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
    lambda_random_latent: float = 1.0,
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
        lambda_random_latent=lambda_random_latent,
    )


class TestRandomLatentDisabled:
    def test_zero_loss_and_no_extra_forward_passes(self):
        compressor = _FakeCompressorV100()
        expander = _FakeExpanderV100()
        trainer = _build_trainer(compressor, expander, train_random_latent=False)

        result = trainer._train_generator(torch.randn(2, *_IMG_SHAPE), global_step=0)

        assert result["random_latent_loss"] == 0.0
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
        assert result["random_latent_loss"] == 0.0


class TestRandomLatentEnabled:
    def test_loss_nonzero_and_present_with_double_forward_passes(self):
        torch.manual_seed(0)
        compressor = _FakeCompressorV100()
        expander = _FakeExpanderV100()
        trainer = _build_trainer(compressor, expander, train_random_latent=True)

        result = trainer._train_generator(torch.randn(2, *_IMG_SHAPE), global_step=0)

        assert "random_latent_loss" in result
        assert result["random_latent_loss"] > 0.0
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
        assert torch.isfinite(torch.tensor(result["random_latent_loss"]))

    def test_final_return_includes_detached_float_value(self):
        torch.manual_seed(0)
        compressor = _FakeCompressorV100()
        expander = _FakeExpanderV100()
        trainer = _build_trainer(compressor, expander, train_random_latent=True)

        result = trainer._train_generator(torch.randn(2, *_IMG_SHAPE), global_step=0)

        assert isinstance(result["random_latent_loss"], float)
