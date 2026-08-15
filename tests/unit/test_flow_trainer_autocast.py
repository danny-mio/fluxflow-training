"""Regression tests for FlowTrainer mixed-precision compute (Bug B).

Root cause: no ``torch.autocast``/``accelerator.autocast()`` context manager
existed anywhere in flow_trainer.py -- ``Accelerator(mixed_precision="fp16")``
only auto-casts forward passes for objects passed through
``accelerator.prepare()``, and FlowTrainer's models never go through
``.prepare()``. So ``use_fp16`` only activated overhead with zero actual
precision reduction.

Fix: wrap the compressor and flow_processor forward passes in
``self.accelerator.autocast():``. Companion to test_vae_trainer_autocast.py
(see that file for the AcceleratorState-singleton-reset rationale).
"""

import contextlib

import pytest
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import AcceleratorState

from fluxflow_training.training.flow_trainer import FlowTrainer

_VAE_DIM = 8
_CONTEXT_DIMS = 0
_TOKEN_COUNT = 4


@pytest.fixture(autouse=True)
def _reset_accelerator_state():
    AcceleratorState._reset_state(reset_partial_state=True)
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


class _DtypeCapturingCompressor(nn.Module):
    """Mock VAE compressor returning a realistic latent packet; records dtype."""

    def __init__(self, vae_dim=_VAE_DIM, context_dims=_CONTEXT_DIMS, token_count=_TOKEN_COUNT):
        super().__init__()
        total_dim = vae_dim + context_dims
        self.total_dim = total_dim
        self.token_count = token_count
        self.use_gradient_checkpointing = False
        self.proj = nn.Linear(total_dim, total_dim)
        self.last_forward_dtype: torch.dtype | None = None

    def get_context_dims(self):
        return _CONTEXT_DIMS

    def forward(self, imgs):
        B = imgs.shape[0]
        base = torch.randn(B, self.token_count + 1, self.total_dim, device=imgs.device)
        out = self.proj(base)
        self.last_forward_dtype = out.dtype
        out = out.clone()
        out[:, -1, :] = 0.0
        out[:, -1, 0] = 4 / 64.0
        out[:, -1, 1] = 4 / 64.0
        return out.detach()


class _DtypeCapturingFlow(nn.Module):
    """Tiny real flow model covering all dims -- produces real gradients, records dtype."""

    def __init__(self, total_dim):
        super().__init__()
        self.proj = nn.Linear(total_dim, total_dim)
        self.last_forward_dtype: torch.dtype | None = None

    def forward(self, packet, text_emb, t):
        img_seq = packet[:, :-1, :]
        hw_vec = packet[:, -1:, :]
        out = self.proj(img_seq)
        self.last_forward_dtype = out.dtype
        return torch.cat([out, hw_vec], dim=1)


class _SimpleTextEncoder(nn.Module):
    def forward(self, ids, attention_mask=None):
        return torch.randn(ids.shape[0], 64, device=ids.device)


def _build_trainer(accelerator, device: str = "cpu") -> FlowTrainer:
    total_dim = _VAE_DIM + _CONTEXT_DIMS
    flow = _DtypeCapturingFlow(total_dim).to(device)
    compressor = _DtypeCapturingCompressor().to(device)
    text_encoder = _SimpleTextEncoder().to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    return FlowTrainer(
        flow_processor=flow,
        text_encoder=text_encoder,
        compressor=compressor,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clip_norm=1.0,
        num_train_timesteps=100,
        accelerator=accelerator,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 autocast requires CUDA")
class TestAutocastEnabledWhenFp16:
    def test_compressor_and_flow_processor_run_in_fp16(self):
        accelerator = Accelerator(cpu=False, mixed_precision="fp16")
        trainer = _build_trainer(accelerator, device=str(accelerator.device))

        B = 2
        real_imgs = torch.randn(B, 3, 32, 32, device=accelerator.device)
        input_ids = torch.randint(0, 100, (B, 8), device=accelerator.device)
        attn = torch.ones(B, 8, device=accelerator.device)

        trainer.train_step(real_imgs, input_ids, attn, global_step=0)

        assert trainer.compressor.last_forward_dtype == torch.float16
        assert trainer.flow_processor.last_forward_dtype == torch.float16


class TestAutocastNotEnteredWhenFp16Disabled:
    def test_compressor_and_flow_processor_stay_fp32(self):
        accelerator = Accelerator(cpu=True, mixed_precision="no")
        trainer = _build_trainer(accelerator)

        B = 2
        real_imgs = torch.randn(B, 3, 32, 32)
        input_ids = torch.randint(0, 100, (B, 8))
        attn = torch.ones(B, 8)

        trainer.train_step(real_imgs, input_ids, attn, global_step=0)

        assert trainer.compressor.last_forward_dtype == torch.float32
        assert trainer.flow_processor.last_forward_dtype == torch.float32


class TestAutocastContextIsEntered:
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

        B = 2
        real_imgs = torch.randn(B, 3, 32, 32)
        input_ids = torch.randint(0, 100, (B, 8))
        attn = torch.ones(B, 8)
        trainer.train_step(real_imgs, input_ids, attn, global_step=0)

        assert calls["enter"] >= 1, "accelerator.autocast() must be entered during train_step"
        assert calls["enter"] == calls["exit"]
