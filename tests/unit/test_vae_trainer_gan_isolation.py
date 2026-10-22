"""G loss must not leak grads into D; D loss has no unconditional-fake path / dead ctx predictor."""

import torch

from fluxflow_training.training import vae_trainer
from tests.unit.test_vae_trainer_gradient_accumulation import _IMG_SHAPE, _build_trainer


def _imgs() -> torch.Tensor:
    return torch.randn(2, *_IMG_SHAPE)


def _d_grads(trainer):
    return [None if p.grad is None else p.grad.clone() for p in trainer.discriminator.parameters()]


def test_generator_leaves_discriminator_grads_none():
    trainer = _build_trainer(use_gan=True)
    trainer._train_generator(_imgs(), global_step=0)
    assert all(p.grad is None for p in trainer.discriminator.parameters())


def test_generator_restores_requires_grad_flags():
    trainer = _build_trainer(use_gan=True)
    params = list(trainer.discriminator.parameters())
    params[0].requires_grad_(False)
    trainer._train_generator(_imgs(), global_step=0)
    assert [p.requires_grad for p in params] == [False] + [True] * (len(params) - 1)


def test_generator_restores_requires_grad_on_error():
    trainer = _build_trainer(use_gan=True)
    orig = trainer.discriminator.forward
    trainer.discriminator.forward = lambda *a, **k: (_ for _ in ()).throw(ValueError("x"))
    try:
        trainer._train_generator(_imgs(), global_step=0)
    except ValueError:
        pass
    trainer.discriminator.forward = orig
    assert all(p.requires_grad for p in trainer.discriminator.parameters())


def test_accumulation_d_grads_come_only_from_d_loss():
    trainer = _build_trainer(use_gan=True, gradient_accumulation_steps=2)
    trainer._train_discriminator(_imgs(), global_step=1, fake_imgs=_imgs())
    before = _d_grads(trainer)
    assert any(g is not None for g in before)
    trainer._train_generator(_imgs(), global_step=1)
    after = _d_grads(trainer)
    for b, a in zip(before, after):
        assert (b is None) == (a is None)
        if b is not None:
            assert torch.equal(b, a)


def test_discriminator_uses_single_hinge_and_no_uncond_expander_call():
    trainer = _build_trainer(use_gan=True)
    calls = []
    orig = trainer.expander.forward
    trainer.expander.forward = lambda packed, use_context=True: (
        calls.append(use_context) or orig(packed, use_context=use_context)
    )
    hinge_calls = []
    real_hinge = vae_trainer.d_hinge_loss
    vae_trainer.d_hinge_loss = lambda *a: hinge_calls.append(1) or real_hinge(*a)
    try:
        trainer._train_discriminator(_imgs(), global_step=1, fake_imgs=_imgs())
    finally:
        vae_trainer.d_hinge_loss = real_hinge
    assert len(calls) == 0  # D reuses the G-step reconstruction
    assert len(hinge_calls) == 1


def test_discriminator_never_runs_context_predictor():
    trainer = _build_trainer(use_gan=True)
    trainer.context_predictor = None  # would raise if touched
    trainer._train_discriminator(_imgs(), global_step=1, fake_imgs=_imgs())
