"""Regression test for the v0.10.0 bezier_reg wiring in VAETrainer._train_generator.

Plan: plans/01-vae-v0.10.0-improvements.md, Fix 2. ``bezier_reg`` was a dead
logging stub (``gen_losses.get("bezier_reg", 0.0)`` at vae_trainer.py:925 --
nothing ever computed or set the key). This test proves the fix by deliberately
inverting a control point on a real ``mu_activation`` instance and checking that
backward() through ``_train_generator``'s ``total_loss`` actually reaches it.

A bare "gen_losses['bezier_reg'] is nonzero" assertion would be a weak test --
default-init models legitimately report ~0.0 too, and it wouldn't catch the reg
being computed but never added to total_loss, or applied to the wrong module.
Checking real gradient flow on the specific mis-ordered control points catches
both classes of mis-wiring: the fixture below never routes ``mu_activation``
into the compressor's forward pass, so the ONLY way p0/p1 can receive a
gradient is via the bezier_reg term actually being summed into total_loss.
"""

from unittest.mock import MagicMock

import torch
import torch.nn as nn
from fluxflow.models.activations import TrainableBezier, WideTrainableBezier

from fluxflow_training.training.vae_trainer import VAETrainer

_TOKEN_DIM = 8
_IMG_SHAPE = (3, 8, 8)


class _FakeCompressorWithBezier(nn.Module):
    """Minimal compressor stub exposing real mu_activation/logvar_activation.

    Mirrors ``_FakeCompressor`` in test_vae_trainer_gradient_accumulation.py,
    plus real ``TrainableBezier``/``WideTrainableBezier`` submodules matching
    the v100 compressor's actual attribute names (``models/v100/vae.py:316-326``).
    Deliberately NOT wired into the mu/logvar computation below -- p0..p3
    should have zero other path to gradient, so any nonzero .grad on them can
    only come from the bezier_reg term in _train_generator.
    """

    def __init__(self, in_channels: int, token_dim: int = 8, n_tokens: int = 3):
        super().__init__()
        self.token_dim = token_dim
        self.n_tokens = n_tokens
        self.proj = nn.Linear(in_channels, token_dim)
        self.use_gradient_checkpointing = False
        self.mu_activation = TrainableBezier(shape=(4, 2, 2))
        self.logvar_activation = WideTrainableBezier(shape=(4, 2, 2))

    def forward(self, x, training=False):
        B = x.size(0)
        feat = x.mean(dim=(2, 3))  # [B, C]
        pooled = self.proj(feat)  # [B, token_dim]
        packed = pooled.unsqueeze(1).expand(B, self.n_tokens + 1, self.token_dim).contiguous()
        mu = torch.zeros(B, 4, 2, 2)
        logvar = torch.zeros(B, 4, 2, 2)
        return packed, mu, logvar

    def get_context_dims(self):
        return self.token_dim


class _FakeExpander(nn.Module):
    """Minimal expander stub with a real trainable param feeding the output."""

    def __init__(self, img_shape: tuple):
        super().__init__()
        self.img_shape = img_shape
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, packed, use_context=True):
        B = packed.size(0)
        C, H, W = self.img_shape
        base = packed[:, 0, 0].view(B, 1, 1, 1).expand(B, C, H, W)
        return base * self.scale


class _PlainAccelerator:
    """No-AMP accelerator double matching production with mixed_precision='no'."""

    def __init__(self):
        self.scaler = None

    def backward(self, loss):
        loss.backward()

    def unscale_gradients(self, optimizer=None):
        pass

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

    def autocast(self):
        import contextlib

        return contextlib.nullcontext()


def _build_trainer(compressor) -> VAETrainer:
    expander = _FakeExpander(_IMG_SHAPE)
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
        train_reconstruction=True,
        train_kl=False,
        train_colorstats=False,
        train_histogram=False,
        train_contrast=False,
        train_coarseness=False,
        train_ctx_aux=False,
        use_lpips=False,
        use_gan=False,
        ctx_input_dim=_TOKEN_DIM,
        context_channels=2,
        context_height=2,
        context_width=2,
        r1_interval=1000,
        accelerator=_PlainAccelerator(),
        gradient_accumulation_steps=1,
    )


class TestBezierRegGradientFlow:
    def test_inverted_mu_activation_control_point_receives_gradient(self):
        """Invert p0>p1 on mu_activation; backward through total_loss must reach it."""
        compressor = _FakeCompressorWithBezier(in_channels=_IMG_SHAPE[0], token_dim=_TOKEN_DIM)
        with torch.no_grad():
            compressor.mu_activation.p0.fill_(0.9)
            compressor.mu_activation.p1.fill_(0.1)
        trainer = _build_trainer(compressor)

        result = trainer._train_generator(torch.randn(2, *_IMG_SHAPE), global_step=0)

        assert result["_optimizer_stepped"] is True
        assert compressor.mu_activation.p0.grad is not None
        assert compressor.mu_activation.p1.grad is not None
        assert compressor.mu_activation.p0.grad.abs().sum().item() > 0
        assert compressor.mu_activation.p1.grad.abs().sum().item() > 0

    def test_inverted_logvar_activation_control_point_receives_gradient(self):
        """Same check on logvar_activation (WideTrainableBezier)."""
        compressor = _FakeCompressorWithBezier(in_channels=_IMG_SHAPE[0], token_dim=_TOKEN_DIM)
        with torch.no_grad():
            compressor.logvar_activation.p2.fill_(-1.0)
            compressor.logvar_activation.p3.fill_(-5.0)
        trainer = _build_trainer(compressor)

        trainer._train_generator(torch.randn(2, *_IMG_SHAPE), global_step=0)

        assert compressor.logvar_activation.p2.grad is not None
        assert compressor.logvar_activation.p3.grad is not None
        assert compressor.logvar_activation.p2.grad.abs().sum().item() > 0
        assert compressor.logvar_activation.p3.grad.abs().sum().item() > 0

    def test_default_monotonic_init_still_reports_zero_bezier_reg(self):
        """Default (monotonic) init: bezier_reg key present and ~0.0 -- a bare
        nonzero check alone would be a weak test since this legitimately holds."""
        compressor = _FakeCompressorWithBezier(in_channels=_IMG_SHAPE[0], token_dim=_TOKEN_DIM)
        trainer = _build_trainer(compressor)

        result = trainer._train_generator(torch.randn(2, *_IMG_SHAPE), global_step=0)

        assert "bezier_reg" in result
        assert abs(result["bezier_reg"]) < 1e-6

    def test_bezier_reg_key_populated_in_mid_accumulation_return(self):
        compressor = _FakeCompressorWithBezier(in_channels=_IMG_SHAPE[0], token_dim=_TOKEN_DIM)
        with torch.no_grad():
            compressor.mu_activation.p1.fill_(0.9)
            compressor.mu_activation.p2.fill_(0.1)
        expander = _FakeExpander(_IMG_SHAPE)
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
            train_reconstruction=True,
            train_kl=False,
            train_colorstats=False,
            train_histogram=False,
            train_contrast=False,
            train_coarseness=False,
            train_ctx_aux=False,
            use_lpips=False,
            use_gan=False,
            ctx_input_dim=_TOKEN_DIM,
            context_channels=2,
            context_height=2,
            context_width=2,
            r1_interval=1000,
            accelerator=_PlainAccelerator(),
            gradient_accumulation_steps=2,
        )

        result = trainer._train_generator(torch.randn(2, *_IMG_SHAPE), global_step=0)

        assert result["_optimizer_stepped"] is False
        assert "bezier_reg" in result
        assert result["bezier_reg"] > 0.0
