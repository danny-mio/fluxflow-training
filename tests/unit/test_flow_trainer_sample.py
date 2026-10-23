"""Flow targets are built from SAMPLED latents (v100); legacy compressors still work."""

import torch
from accelerate import Accelerator

from tests.unit.test_flow_trainer_autocast import _DtypeCapturingCompressor, _build_trainer


class _SampleAware(_DtypeCapturingCompressor):
    seen = "unset"

    def forward(self, imgs, training=False, sample=None):
        type(self).seen = sample
        return super().forward(imgs)


def _step(trainer):
    trainer.train_step(
        torch.randn(2, 3, 32, 32), torch.randint(0, 100, (2, 8)), torch.ones(2, 8), global_step=0
    )


def test_train_step_requests_sampled_latents():
    trainer = _build_trainer(Accelerator(cpu=True, mixed_precision="no"))
    trainer.compressor = _SampleAware()
    _step(trainer)
    assert _SampleAware.seen is True


def test_train_step_legacy_compressor_without_sample():
    trainer = _build_trainer(Accelerator(cpu=True, mixed_precision="no"))
    _step(trainer)  # forward(imgs) only; must not raise TypeError
