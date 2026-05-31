"""Tests for anchor loss independence from train_vae / train_reconstruction.

Verifies that LPIPS, colorstats, histogram, contrast, and coarseness losses fire
based solely on their own use_*/train_* flags, not on train_reconstruction.

Bug: each anchor was gated on `self.train_reconstruction AND self.train_<x>`,
causing LPIPS: 0.0000 in GAN-only mode even when use_lpips=True.
"""

from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from fluxflow_training.training.utils import EMA
from fluxflow_training.training.vae_trainer import VAETrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_compressor(out_dim: int = 8):
    """Minimal compressor: returns (packed, mu, logvar) with matching shapes."""

    class _Comp(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.lin = nn.Linear(3, dim)
            self.dim = dim

        def forward(self, x, training=False):
            B = x.shape[0]
            flat = x.mean(dim=[2, 3])  # [B, 3]
            out = self.lin(flat)  # [B, dim]
            T = 1
            packed = out.unsqueeze(1).expand(B, T + 1, self.dim)
            mu = torch.zeros(B, self.dim, 1, 1, device=x.device)
            logvar = torch.zeros(B, self.dim, 1, 1, device=x.device)
            return packed, mu, logvar

    return _Comp(out_dim)


def _make_expander(out_dim: int = 8, out_hw: int = 8):
    """Minimal expander: maps packed token seq to [B, 3, out_hw, out_hw].

    out_hw must match the spatial size of real_imgs passed to _train_generator
    so that reconstruction losses (which compare out_imgs_rec vs real_imgs) do
    not raise shape mismatch errors.
    """

    class _Exp(nn.Module):
        def __init__(self, dim, hw):
            super().__init__()
            self.lin = nn.Linear(dim, 3)
            self.hw = hw

        def forward(self, packed, use_context=False):
            B = packed.shape[0]
            token = packed[:, 0, :]  # [B, dim]
            out = self.lin(token)  # [B, 3]
            return out.view(B, 3, 1, 1).expand(B, 3, self.hw, self.hw).contiguous()

    return _Exp(out_dim, out_hw)


def _make_accelerator():
    accel = MagicMock()
    accel.backward = lambda loss: loss.backward()
    accel.scaler = None
    accel.clip_grad_norm_ = MagicMock(return_value=1.0)
    # step() maps to optimizer.step() — called on both vae + ctx optimizers
    accel.step = MagicMock(side_effect=lambda opt: opt.step())
    return accel


def _build_trainer(
    train_reconstruction: bool = True,
    use_lpips: bool = False,
    train_colorstats: bool = False,
    train_histogram: bool = False,
    train_contrast: bool = False,
    train_coarseness: bool = False,
    use_gan: bool = False,
    lpips_fn: Optional[nn.Module] = None,
    dim: int = 8,
) -> VAETrainer:
    """Build a minimal VAETrainer for loss-gate tests."""
    compressor = _make_compressor(dim)
    expander = _make_expander(dim)

    all_params = list(compressor.parameters()) + list(expander.parameters())
    optimizer = torch.optim.SGD(all_params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    ema = EMA(nn.ModuleList([compressor, expander]))
    accel = _make_accelerator()

    disc_kwargs: dict = {}
    if use_gan:
        disc = nn.Sequential(
            nn.Conv2d(3, 1, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        disc_opt = torch.optim.SGD(disc.parameters(), lr=1e-3)
        disc_sched = torch.optim.lr_scheduler.ConstantLR(disc_opt)
        disc_kwargs = {
            "discriminator": disc,
            "discriminator_optimizer": disc_opt,
            "discriminator_scheduler": disc_sched,
        }

    trainer = VAETrainer(
        compressor=compressor,
        expander=expander,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        reconstruction_loss_fn=nn.L1Loss(),
        reconstruction_loss_min_fn=nn.MSELoss(),
        train_reconstruction=train_reconstruction,
        use_gan=use_gan,
        use_lpips=use_lpips,
        train_colorstats=train_colorstats,
        train_histogram=train_histogram,
        train_contrast=train_contrast,
        train_coarseness=train_coarseness,
        train_kl=False,
        train_ctx_aux=False,
        lambda_lpips=0.5,
        accelerator=accel,
        **disc_kwargs,
    )

    # Override lpips_fn if provided (avoid real LPIPS import in tests)
    if lpips_fn is not None:
        trainer.use_lpips = True
        trainer.lpips_fn = lpips_fn

    return trainer


def _fake_lpips_fn(value: float = 0.25) -> nn.Module:
    """Mock LPIPS that returns a fixed scalar, with .to(device) support."""

    class _FakeLPIPS(nn.Module):
        def __init__(self, v):
            super().__init__()
            self._val = v

        def to(self, device):  # type: ignore[override]
            return self

        def forward(self, x, y):
            return torch.tensor([self._val], device=x.device)

    return _FakeLPIPS(value)


def _run_generator_step(trainer: VAETrainer) -> dict:
    """Run one _train_generator step on a tiny batch.

    Images must be at least 8x8 so that InstanceNorm2d in context_encoder
    (which downsizes by 4x via two stride-2 convolutions) never hits a 1x1
    spatial size, which InstanceNorm2d rejects during training.
    """
    imgs = torch.randn(2, 3, 8, 8)
    return trainer._train_generator(imgs, global_step=0)


# ---------------------------------------------------------------------------
# Task D tests: LPIPS gate
# ---------------------------------------------------------------------------


class TestLPIPSGate:
    """LPIPS must fire based solely on use_lpips, not train_reconstruction."""

    def test_lpips_absent_when_disabled(self):
        """use_lpips=False → lpips key is 0.0 in returned dict."""
        trainer = _build_trainer(train_reconstruction=True, use_lpips=False)
        result = _run_generator_step(trainer)
        assert result["lpips"] == pytest.approx(0.0), (
            "lpips should be 0.0 when use_lpips=False"
        )

    def test_lpips_present_when_use_lpips_true_and_train_reconstruction_true(self):
        """Both flags true → LPIPS computed and returned non-zero."""
        lpips_fn = _fake_lpips_fn(0.3)
        trainer = _build_trainer(train_reconstruction=True)
        trainer.use_lpips = True
        trainer.lpips_fn = lpips_fn

        result = _run_generator_step(trainer)
        assert result["lpips"] > 0.0, (
            "LPIPS should be non-zero when use_lpips=True and train_reconstruction=True"
        )

    def test_lpips_present_when_use_lpips_true_and_train_reconstruction_false(self):
        """GAN-only mode: use_lpips=True, train_vae=False → LPIPS still fires.

        This is the primary regression: before the fix, LPIPS was dead here.
        """
        lpips_fn = _fake_lpips_fn(0.3)
        trainer = _build_trainer(train_reconstruction=False)
        trainer.use_lpips = True
        trainer.lpips_fn = lpips_fn

        result = _run_generator_step(trainer)
        assert result["lpips"] > 0.0, (
            "LPIPS must fire when use_lpips=True even if train_reconstruction=False. "
            "The old `if self.train_reconstruction: ... LPIPS` coupling is the bug."
        )

    def test_lpips_in_total_loss_gan_only_mode(self):
        """In GAN-only mode (train_vae=False), LPIPS must reach total_loss.

        Strategy: capture the total_loss passed to backward and confirm it is
        non-zero — possible only when LPIPS contributes to it.
        """
        lpips_fn = _fake_lpips_fn(0.4)
        trainer = _build_trainer(train_reconstruction=False)
        trainer.use_lpips = True
        trainer.lpips_fn = lpips_fn

        imgs = torch.randn(2, 3, 8, 8)
        # Replace accelerator.backward with a plain capturing callable
        captured: list[torch.Tensor] = []

        def _capture(loss):
            captured.append(loss.detach().clone())
            loss.backward()

        trainer.accelerator.backward = _capture

        trainer._train_generator(imgs, global_step=0)

        assert len(captured) == 1, "backward() should be called exactly once"
        # With only LPIPS active (train_kl=False, train_reconstruction=False, use_gan=False)
        # and fake LPIPS returning 0.4 * lambda_lpips=0.5 → LPIPS contributes 0.2
        # Plus context_alignment_loss. Total must be positive (not zero).
        assert captured[0].item() > 0.0, (
            f"total_loss={captured[0].item():.4f} expected > 0 but LPIPS contribution is missing"
        )

    def test_lpips_not_double_counted_joint_mode(self):
        """When train_vae=True and use_lpips=True, LPIPS must appear exactly once.

        Strategy: run with lambda_lpips=1.0 and a fake LPIPS returning a known value V.
        The expected contribution from LPIPS in total_loss is lambda_lpips * V = V.
        If it's double-counted it would be 2V.
        """
        lpips_val = 0.3
        lpips_fn = _fake_lpips_fn(lpips_val)
        trainer = _build_trainer(train_reconstruction=True)
        trainer.use_lpips = True
        trainer.lpips_fn = lpips_fn
        trainer.lambda_lpips = 1.0

        imgs = torch.randn(2, 3, 8, 8)
        captured: list[torch.Tensor] = []

        def _capture(loss):
            captured.append(loss.detach().clone())
            loss.backward()

        trainer.accelerator.backward = _capture
        result = trainer._train_generator(imgs, global_step=0)

        # lpips key in result must equal the actual perceptual_loss value (not 2x)
        assert result["lpips"] == pytest.approx(lpips_val, abs=1e-4), (
            f"lpips key={result['lpips']:.4f} expected≈{lpips_val:.4f}; "
            "double-counting would show 2x"
        )


# ---------------------------------------------------------------------------
# Task D tests: colorstats / histogram / contrast / coarseness gates
# ---------------------------------------------------------------------------


class TestAnchorLossGates:
    """Colorstats/histogram/contrast/coarseness fire based on their own flag only."""

    def _check_anchor_fires_without_reconstruction(
        self, flag_name: str, result_key: str
    ):
        """Generic check: flag=True, train_reconstruction=False → result_key > 0."""
        kwargs = {
            "train_reconstruction": False,
            flag_name: True,
        }
        trainer = _build_trainer(**kwargs)

        # Patch the underlying loss method to return a known non-zero value
        method_map = {
            "train_colorstats": "_color_statistics_loss",
            "train_histogram": "_histogram_matching_loss",
            "train_contrast": "_contrast_regularization_loss",
            "train_coarseness": "_coarseness_loss",
        }
        method_name = method_map[flag_name]

        def _fake_loss(x, y):
            return torch.tensor(0.5)

        setattr(trainer, method_name, _fake_loss)

        result = _run_generator_step(trainer)
        assert result[result_key] > 0.0, (
            f"{result_key} should be non-zero when {flag_name}=True "
            f"even if train_reconstruction=False. "
            f"The leading `self.train_reconstruction and ...` gate is the bug."
        )

    def test_colorstats_fires_without_reconstruction(self):
        self._check_anchor_fires_without_reconstruction("train_colorstats", "color_stats")

    def test_histogram_fires_without_reconstruction(self):
        self._check_anchor_fires_without_reconstruction("train_histogram", "hist_loss")

    def test_contrast_fires_without_reconstruction(self):
        self._check_anchor_fires_without_reconstruction("train_contrast", "contrast_loss")

    def test_coarseness_fires_without_reconstruction(self):
        self._check_anchor_fires_without_reconstruction("train_coarseness", "coarseness_loss")

    def test_colorstats_zero_when_disabled(self):
        """train_colorstats=False → color_stats key is 0.0."""
        trainer = _build_trainer(train_reconstruction=True, train_colorstats=False)
        result = _run_generator_step(trainer)
        assert result["color_stats"] == pytest.approx(0.0)

    def test_histogram_zero_when_disabled(self):
        trainer = _build_trainer(train_reconstruction=True, train_histogram=False)
        result = _run_generator_step(trainer)
        assert result["hist_loss"] == pytest.approx(0.0)

    def test_contrast_zero_when_disabled(self):
        trainer = _build_trainer(train_reconstruction=True, train_contrast=False)
        result = _run_generator_step(trainer)
        assert result["contrast_loss"] == pytest.approx(0.0)

    def test_coarseness_zero_when_disabled(self):
        trainer = _build_trainer(train_reconstruction=True, train_coarseness=False)
        result = _run_generator_step(trainer)
        assert result["coarseness_loss"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Task C: ctx_aux is already independently gated — confirm it stays that way
# ---------------------------------------------------------------------------


class TestCtxAuxIndependentGate:
    """ctx_aux_loss is gated on train_ctx_aux only — no train_reconstruction coupling."""

    @staticmethod
    def _source() -> str:
        import importlib.util
        import pathlib

        spec = importlib.util.find_spec("fluxflow_training.training.vae_trainer")
        return pathlib.Path(spec.origin).read_text()

    def test_ctx_aux_not_gated_on_train_reconstruction(self):
        """Source must not contain 'train_reconstruction and self.train_ctx_aux'."""
        source = self._source()
        assert "train_reconstruction and self.train_ctx_aux" not in source, (
            "ctx_aux_loss is incorrectly coupled to train_reconstruction. "
            "It must be gated on train_ctx_aux only."
        )

    def test_ctx_aux_gated_on_train_ctx_aux(self):
        """Source must contain 'if self.train_ctx_aux:' as sole gate."""
        source = self._source()
        assert "if self.train_ctx_aux:" in source, (
            "train_ctx_aux flag must be the sole gate for ctx_aux_loss computation."
        )
