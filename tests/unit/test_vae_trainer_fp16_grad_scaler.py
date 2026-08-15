"""Regression tests for the fp16 GradScaler crash in VAETrainer.

Real crash (training.use_fp16: true, step "vae_recon_warmup_coco", gan_training:
false, gradient_accumulation_steps: 16)::

    File ".../vae_trainer.py", line 1519, in _train_generator
        self.accelerator.scaler.unscale_(self.optimizer)
    RuntimeError: unscale_() has already been called on this optimizer since the
    last update().

Root cause: ``_train_generator`` called ``self.accelerator.scaler.unscale_(self.optimizer)``
manually (to NaN-guard gradients before clipping) but the file never called
``scaler.update()`` anywhere. VAETrainer's optimizers are plain ``torch.optim``
instances -- the ``PipelineOrchestrator`` training path only ever calls
``accelerator.prepare()`` on the dataloader (see
``pipeline_orchestrator.py::_create_step_optimizers``, which builds optimizers via
the ``create_optimizer`` factory directly) -- so nothing else in the stack ever
resets the ``GradScaler``'s per-optimizer "already unscaled" stage between calls.
The first ``_train_generator`` call unscales fine; the very next call (next
resolution in the "train on all resolutions" loop, or next batch) reuses the same
optimizer identity and crashes, because ``accelerator.clip_grad_norm_()`` right
after it *also* tries to unscale internally (double-unscale within one call is the
other half of this same defect -- see ``_FakeAccelerator.clip_grad_norm_`` below,
which mirrors real Accelerate 1.14.0's ``Accelerator.clip_grad_norm_`` ->
``unscale_gradients()`` chain).

``_train_discriminator`` has the same class of bug in a quieter form: it never
unscales at all before ``discriminator_optimizer.step()``, so under fp16 it would
silently apply gradients still multiplied by the GradScaler's growth factor
(>= 2**16) the first time ``gan_training: true`` runs under fp16 -- no crash, but
immediate divergence. Fixed the same way: unscale once via
``accelerator.unscale_gradients()`` before stepping.

These tests use a REAL ``torch.amp.GradScaler`` (not a mock) so the actual PyTorch
"already been called" ``RuntimeError`` is what (dis)proves the fix, driven through
a lightweight fake ``Accelerator`` double that reproduces exactly the two pieces of
real Accelerate behavior that matter here: (a) ``clip_grad_norm_`` internally calls
``unscale_gradients()``, and (b) that registry is empty for VAETrainer's optimizers
in production (never prepared), matching ``optimizer=None`` unscale calls being a
no-op while an explicit ``unscale_gradients(optimizer)`` call still works.
"""

import contextlib
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from fluxflow.models.activations import TrainableBezier, WideTrainableBezier

from fluxflow_training.training.vae_trainer import VAETrainer


class _FakeCompressor(nn.Module):
    """Minimal compressor stub with real trainable params (needs real gradients)."""

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


class _FakeDiscriminator(nn.Module):
    """Minimal patch discriminator stub with a ctx_proj for dim validation."""

    def __init__(self, ctx_dim: int, img_channels: int = 3):
        super().__init__()
        self.ctx_proj = nn.Linear(ctx_dim, 4)
        self.ctx_dim = ctx_dim
        self.conv = nn.Conv2d(img_channels, 1, 3, padding=1)

    def forward(self, x, ctx=None):
        return self.conv(x)


_TOKEN_DIM = 8
_IMG_SHAPE = (3, 8, 8)


class _FakeAccelerator:
    """Reproduces the slice of real Accelerate 1.14.0 semantics relevant to this
    bug, backed by a REAL GradScaler.

    - ``self._optimizers`` mirrors ``Accelerator._optimizers``: only populated by
      ``accelerator.prepare()``, which VAETrainer's optimizers never go through in
      the ``PipelineOrchestrator`` path. Left empty, exactly like production.
    - ``clip_grad_norm_`` mirrors ``Accelerator.clip_grad_norm_``'s non-FSDP branch:
      unconditionally calls ``unscale_gradients()`` (optimizer=None) before
      clipping.
    - No ``.step()`` attribute, matching the real ``Accelerator`` -- forces
      VAETrainer's ``accelerator_step`` fallback to call ``optimizer.step()``
      directly, exactly like production.
    - ``autocast()`` is a real no-op context manager (``nullcontext``): these
      tests exercise GradScaler unscale/step bookkeeping, not autocast dtype
      behavior (that's covered by test_vae_trainer_autocast.py), but VAETrainer
      now calls ``self.accelerator.autocast()`` around forward passes, so the
      double must support it.
    """

    def __init__(self):
        self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        self._optimizers: list = []
        self.unscale_calls: list = []

    def autocast(self):
        return contextlib.nullcontext()

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def unscale_gradients(self, optimizer=None):
        targets = self._optimizers if optimizer is None else [optimizer]
        for opt in targets:
            self.unscale_calls.append(opt)
            self.scaler.unscale_(opt)

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        self.unscale_gradients()  # optimizer=None, real Accelerate's default
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)


def _build_trainer(accelerator, use_gan: bool = False) -> VAETrainer:
    compressor = _FakeCompressor(in_channels=_IMG_SHAPE[0], token_dim=_TOKEN_DIM)
    expander = _FakeExpander(_IMG_SHAPE)
    opt = torch.optim.SGD(list(compressor.parameters()) + list(expander.parameters()), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

    kwargs = dict(
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
        ctx_input_dim=_TOKEN_DIM,
        context_channels=2,
        context_height=2,
        context_width=2,
        r1_interval=1000,
        accelerator=accelerator,
    )
    if use_gan:
        discriminator = _FakeDiscriminator(ctx_dim=_TOKEN_DIM)
        discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=1e-3)
        kwargs.update(
            use_gan=True,
            discriminator=discriminator,
            discriminator_optimizer=discriminator_optimizer,
            discriminator_scheduler=MagicMock(),
        )
    else:
        kwargs["use_gan"] = False

    return VAETrainer(**kwargs)


class TestFp16GradScalerDoubleUnscale:
    """Reproduces the real crash: two consecutive train_step calls under fp16
    mixed precision, gan_training: false -- matches the bug report exactly."""

    def test_two_consecutive_train_steps_do_not_raise_under_fp16(self):
        accelerator = _FakeAccelerator()
        trainer = _build_trainer(accelerator, use_gan=False)

        # Must not raise. Previously the second call hit:
        # "RuntimeError: unscale_() has already been called on this optimizer
        # since the last update()."
        trainer.train_step(torch.randn(2, *_IMG_SHAPE), global_step=1)
        trainer.train_step(torch.randn(2, *_IMG_SHAPE), global_step=1)

    def test_three_consecutive_train_steps_do_not_raise_under_fp16(self):
        """Simulates the multi-resolution training loop in pipeline_orchestrator.py,
        which calls train_step multiple times for the same global_step."""
        accelerator = _FakeAccelerator()
        trainer = _build_trainer(accelerator, use_gan=False)

        for _ in range(3):
            trainer.train_step(torch.randn(2, *_IMG_SHAPE), global_step=1)

    def test_optimizer_is_unscaled_exactly_once_per_call(self):
        """Guards against a fix that merely swallows the crash: the VAE optimizer
        must be unscaled exactly once per train_step call, not zero (silently
        stepping on still-scaled gradients) or two-plus (the original bug)."""
        accelerator = _FakeAccelerator()
        trainer = _build_trainer(accelerator, use_gan=False)

        trainer.train_step(torch.randn(2, *_IMG_SHAPE), global_step=1)

        assert accelerator.unscale_calls.count(trainer.optimizer) == 1


class TestFp16GradScalerDiscriminatorPath:
    """_train_discriminator had the same bug pattern in quieter form: it stepped
    discriminator_optimizer directly with no unscale_ call at all under fp16."""

    def test_two_consecutive_train_steps_do_not_raise_with_gan_under_fp16(self):
        accelerator = _FakeAccelerator()
        trainer = _build_trainer(accelerator, use_gan=True)

        trainer.train_step(torch.randn(2, *_IMG_SHAPE), global_step=1)
        trainer.train_step(torch.randn(2, *_IMG_SHAPE), global_step=2)

    def test_discriminator_optimizer_gradients_are_unscaled_before_step(self):
        accelerator = _FakeAccelerator()
        trainer = _build_trainer(accelerator, use_gan=True)

        trainer.train_step(torch.randn(2, *_IMG_SHAPE), global_step=1)

        assert trainer.discriminator_optimizer in accelerator.unscale_calls
