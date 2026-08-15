"""Regression tests for VAETrainer mixed-precision compute (Bug B).

Root cause: no ``torch.autocast``/``accelerator.autocast()`` context manager
existed anywhere in vae_trainer.py -- ``Accelerator(mixed_precision="fp16")``
only auto-casts forward passes for objects passed through
``accelerator.prepare()``, and VAETrainer's models never go through
``.prepare()`` (see ``PipelineOrchestrator._create_step_optimizers``). So
``use_fp16`` only activated GradScaler bookkeeping overhead with zero actual
precision reduction.

Fix: wrap the compressor/expander/discriminator forward passes in
``self.accelerator.autocast():``. These tests spy on ``accelerator.autocast``
being entered during the expensive forward passes, and confirm the dtype of
an activation actually produced inside that context is float16 when fp16 is
enabled (using a REAL ``torch.autocast`` from a real Accelerate ``Accelerator``
instance -- not a mock -- since a mock can't tell us the actual compute dtype).

``accelerate.state.AcceleratorState`` is a process-wide singleton, so each test
that constructs a real ``Accelerator`` resets it first via the private
``_reset_state()`` hook Accelerate's own test suite uses for this exact
purpose -- otherwise the second ``Accelerator(...)`` call in the process raises
"AcceleratorState has already been initialized".
"""

import contextlib

import pytest
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from fluxflow.models.activations import TrainableBezier, WideTrainableBezier

from fluxflow_training.training.vae_trainer import VAETrainer

_TOKEN_DIM = 8
_IMG_SHAPE = (3, 8, 8)


@pytest.fixture(autouse=True)
def _reset_accelerator_state():
    AcceleratorState._reset_state(reset_partial_state=True)
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


class _DtypeCapturingCompressor(nn.Module):
    """Records the dtype its internal Linear activation runs at."""

    def __init__(self, in_channels: int, token_dim: int = 8, n_tokens: int = 3):
        super().__init__()
        self.token_dim = token_dim
        self.n_tokens = n_tokens
        self.proj = nn.Linear(in_channels, token_dim)
        self.use_gradient_checkpointing = False
        self.last_forward_dtype: torch.dtype | None = None
        self.mu_activation = TrainableBezier(shape=(4, 2, 2))
        self.logvar_activation = WideTrainableBezier(shape=(4, 2, 2))

    def forward(self, x, training=False):
        B = x.size(0)
        feat = x.mean(dim=(2, 3))
        pooled = self.proj(feat)
        self.last_forward_dtype = pooled.dtype
        packed = pooled.unsqueeze(1).expand(B, self.n_tokens + 1, self.token_dim).contiguous()
        mu = torch.zeros(B, 4, 2, 2, device=x.device)
        logvar = torch.zeros(B, 4, 2, 2, device=x.device)
        return packed, mu, logvar

    def get_context_dims(self):
        return self.token_dim


class _DtypeCapturingExpander(nn.Module):
    """Records the dtype its internal computation runs at."""

    def __init__(self, img_shape: tuple):
        super().__init__()
        self.img_shape = img_shape
        C, H, W = img_shape
        # A real nn.Linear (matmul-based, like the real SPADE decoder's conv/linear
        # layers) so autocast's op-level fp16 casting actually applies -- plain
        # elementwise ops (e.g. tensor * Parameter) follow ordinary dtype
        # promotion under autocast, not forced fp16, so they're not representative.
        self.proj = nn.Linear(1, C * H * W)
        self.last_forward_dtype: torch.dtype | None = None

    def forward(self, packed, use_context=True):
        B = packed.size(0)
        C, H, W = self.img_shape
        base = packed[:, 0, 0].view(B, 1)
        out = self.proj(base)
        self.last_forward_dtype = out.dtype
        return out.view(B, C, H, W)


def _build_trainer(accelerator, device: str = "cpu", img_shape: tuple = _IMG_SHAPE) -> VAETrainer:
    compressor = _DtypeCapturingCompressor(in_channels=img_shape[0], token_dim=_TOKEN_DIM).to(
        device
    )
    expander = _DtypeCapturingExpander(img_shape).to(device)
    opt = torch.optim.SGD(list(compressor.parameters()) + list(expander.parameters()), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

    return VAETrainer(
        compressor=compressor,
        expander=expander,
        optimizer=opt,
        scheduler=sched,
        ema=None,
        reconstruction_loss_fn=nn.L1Loss(),
        reconstruction_loss_min_fn=nn.MSELoss(),
        train_reconstruction=True,
        train_kl=False,
        train_colorstats=False,
        train_histogram=False,
        train_contrast=False,
        train_coarseness=False,
        train_ctx_aux=False,
        use_gan=False,
        use_lpips=False,
        ctx_input_dim=_TOKEN_DIM,
        context_channels=2,
        context_height=2,
        context_width=2,
        r1_interval=1000,
        accelerator=accelerator,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 autocast requires CUDA")
class TestAutocastEnabledWhenFp16:
    """With a real Accelerate Accelerator(mixed_precision='fp16'), forward
    passes inside VAETrainer must actually run in half precision."""

    def test_compressor_and_expander_run_in_fp16(self):
        accelerator = Accelerator(cpu=False, mixed_precision="fp16")
        trainer = _build_trainer(accelerator, device=str(accelerator.device))

        trainer.train_step(torch.randn(2, *_IMG_SHAPE, device=accelerator.device), global_step=0)

        assert trainer.compressor.last_forward_dtype == torch.float16
        assert trainer.expander.last_forward_dtype == torch.float16


class TestAutocastNotEnteredWhenFp16Disabled:
    """mixed_precision='no' (the default) must not change compute dtype --
    zero behavior change from before this fix."""

    def test_compressor_and_expander_stay_fp32_on_cpu_no_mixed_precision(self):
        accelerator = Accelerator(cpu=True, mixed_precision="no")
        trainer = _build_trainer(accelerator)

        trainer.train_step(torch.randn(2, *_IMG_SHAPE), global_step=0)

        assert trainer.compressor.last_forward_dtype == torch.float32
        assert trainer.expander.last_forward_dtype == torch.float32


class TestAutocastContextIsEntered:
    """Spy on accelerator.autocast() to confirm it's actually invoked as a
    context manager around the forward passes (idiomatic given this repo's
    existing MagicMock-based accelerator doubles in test_vae_trainer_*.py)."""

    def test_autocast_entered_during_train_step(self):
        real_accelerator = Accelerator(cpu=True, mixed_precision="no")
        calls = {"enter": 0, "exit": 0}
        real_autocast = real_accelerator.autocast

        @contextlib.contextmanager
        def _spy_autocast(*a, **kw):
            calls["enter"] += 1
            with real_autocast(*a, **kw):
                yield
            calls["exit"] += 1

        real_accelerator.autocast = _spy_autocast
        trainer = _build_trainer(real_accelerator)

        trainer.train_step(torch.randn(2, *_IMG_SHAPE), global_step=0)

        assert calls["enter"] >= 1, "accelerator.autocast() must be entered during train_step"
        assert calls["enter"] == calls["exit"]

    def test_autocast_not_required_when_accelerator_is_none(self):
        """accelerator=None (used by some lightweight unit tests that never call
        train_step) must not be forced to support autocast() at construction time."""
        compressor = _DtypeCapturingCompressor(in_channels=_IMG_SHAPE[0], token_dim=_TOKEN_DIM)
        expander = _DtypeCapturingExpander(_IMG_SHAPE)
        opt = torch.optim.SGD(list(compressor.parameters()) + list(expander.parameters()), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

        # Must not raise merely by constructing with accelerator=None.
        VAETrainer(
            compressor=compressor,
            expander=expander,
            optimizer=opt,
            scheduler=sched,
            ema=None,
            reconstruction_loss_fn=nn.L1Loss(),
            reconstruction_loss_min_fn=nn.MSELoss(),
            train_reconstruction=True,
            train_kl=False,
            train_colorstats=False,
            train_histogram=False,
            train_contrast=False,
            train_coarseness=False,
            train_ctx_aux=False,
            use_gan=False,
            use_lpips=False,
            ctx_input_dim=_TOKEN_DIM,
            context_channels=2,
            context_height=2,
            context_width=2,
            r1_interval=1000,
            accelerator=None,
        )


class TestLpipsExcludedFromAutocast:
    """LPIPS/VGG must run in float32 regardless of mixed precision -- numerically
    unstable in fp16, and its own weights are never cast (see lpips.LPIPS.forward,
    which does no internal dtype casting of its inputs)."""

    def test_lpips_receives_float32_input_even_under_fp16(self):
        import lpips as lpips_pkg

        # Real VGG needs >= 5 halvings of spatial size to survive its maxpool
        # stack; the 8x8 fixture used elsewhere in this file is too small.
        img_shape = (3, 64, 64)

        accelerator = Accelerator(cpu=True, mixed_precision="no")
        trainer = _build_trainer(accelerator, img_shape=img_shape)
        trainer.use_lpips = True

        captured_dtypes = []
        real_lpips_forward = lpips_pkg.LPIPS.forward

        def _spy_forward(self, in0, in1, *a, **kw):
            captured_dtypes.append((in0.dtype, in1.dtype))
            return real_lpips_forward(self, in0, in1, *a, **kw)

        trainer.lpips_fn = lpips_pkg.LPIPS(net="vgg", spatial=True)
        trainer.lpips_fn.eval()
        for p in trainer.lpips_fn.parameters():
            p.requires_grad = False
        trainer.lpips_fn.forward = _spy_forward.__get__(trainer.lpips_fn, lpips_pkg.LPIPS)

        trainer.train_step(torch.randn(2, *img_shape), global_step=0)

        assert captured_dtypes, "LPIPS forward was never called"
        for in0_dtype, in1_dtype in captured_dtypes:
            assert in0_dtype == torch.float32
            assert in1_dtype == torch.float32
