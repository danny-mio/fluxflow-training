"""fluxflow-calibrate-latents: set v0.10.0 VAE latent stats on a trained checkpoint.

Encodes ``--num-images`` training images (the config's ``default_dataset`` with its
training resolution mix, via the orchestrator's dataset factory), measures the RAW
per-channel mean/std of the compressor's token rows, writes them into compressor and
expander (``FluxPipeline.set_latent_stats``) and saves the full checkpoint (same key
layout as ``flxflow_final.safetensors``) to ``--out``.

Usage::

    fluxflow-calibrate-latents --checkpoint out/flxflow_final.safetensors \\
        --config config.yaml --out out/flxflow_calibrated.safetensors
"""

import argparse
import dataclasses
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from fluxflow_training.training.latent_calibration import (
    calibrate_models,
    compute_latent_stats,
    context_dims_of,
    format_latent_stats_table,
    has_latent_stats,
)

__all__ = ["compute_latent_stats", "format_latent_stats_table", "main"]


def _parse_cli(argv: Optional[list[str]]) -> argparse.Namespace:
    """Parse ``fluxflow-calibrate-latents`` arguments."""
    parser = argparse.ArgumentParser(
        prog="fluxflow-calibrate-latents",
        description="Calibrate per-channel VAE latent stats (latent_shift/latent_scale) "
        "so the flow sees ~N(0, I) latents.",
    )
    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint (.safetensors)")
    parser.add_argument("--config", required=True, help="Training YAML config used for the run")
    parser.add_argument("--out", required=True, help="Output checkpoint path (.safetensors)")
    parser.add_argument("--num-images", type=int, default=2000, help="Images to encode")
    parser.add_argument("--batch-size", type=int, default=8, help="Calibration batch size")
    parser.add_argument(
        "--force", action="store_true", help="Allow --out to overwrite --checkpoint"
    )
    args = parser.parse_args(argv)
    if args.num_images <= 0 or args.batch_size <= 0:
        parser.error("--num-images and --batch-size must be > 0")
    if Path(args.out).resolve() == Path(args.checkpoint).resolve() and not args.force:
        parser.error("--out equals --checkpoint; pass --force to overwrite it in place")
    return args


def _build_models_and_data(cli: argparse.Namespace) -> tuple[dict, Iterable[Any], Any]:
    """Build models (``initialize_models``) and the calibration dataloader.

    Training args come from ``scripts/train.py parse_args`` with
    ``--config/--model_checkpoint/--batch_size`` so YAML -> args mapping is identical
    to ``fluxflow-train --config``.

    Returns:
        ``(models, dataloader, accelerator)``; the dataloader yields
        ``(list_of_resolution_tensors, input_ids)``.
    """
    import yaml
    from accelerate import Accelerator

    from fluxflow_training.scripts.train import (
        _apply_rocm_workarounds,
        _resolve_mixed_precision,
        initialize_dataloader,
        initialize_models,
        parse_args,
    )
    from fluxflow_training.training.checkpoint_manager import CheckpointManager
    from fluxflow_training.training.pipeline_config import parse_pipeline_config
    from fluxflow_training.training.pipeline_orchestrator import TrainingPipelineOrchestrator

    train_args = parse_args(
        [
            "--config",
            cli.config,
            "--model_checkpoint",
            cli.checkpoint,
            "--batch_size",
            str(cli.batch_size),
        ]
    )
    with open(cli.config, "r") as f:
        config = yaml.safe_load(f) or {}

    use_cuda = torch.cuda.is_available()
    accelerator = Accelerator(
        cpu=not use_cuda,
        mixed_precision=_resolve_mixed_precision(train_args.use_fp16, train_args.precision),
    )
    device = accelerator.device
    _apply_rocm_workarounds(device)

    # Load from the checkpoint's own directory (text_encoder.safetensors sibling
    # override applies exactly as on training resume).
    manager = CheckpointManager(output_dir=Path(cli.checkpoint).resolve().parent)
    models = initialize_models(train_args, config, device, manager)

    pipeline_cfg = (config.get("training") or {}).get("pipeline")
    dataloader: Iterable[Any]
    if pipeline_cfg and pipeline_cfg.get("datasets"):
        pcfg = parse_pipeline_config(pipeline_cfg)
        name = pcfg.default_dataset or next(iter(pcfg.datasets))
        ds_cfg = dataclasses.replace(pcfg.datasets[name], batch_size=cli.batch_size)
        orch = TrainingPipelineOrchestrator(
            pipeline_config=pcfg, checkpoint_manager=manager, accelerator=accelerator
        )
        dataloader, _, _ = orch._create_dataloader_for_dataset(ds_cfg, name, train_args, config)
        print(f"Calibration dataset: {name} ({ds_cfg.type})")
    else:
        dataloader, _, _ = initialize_dataloader(train_args, accelerator)
    return models, dataloader, accelerator


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for ``fluxflow-calibrate-latents``."""
    cli = _parse_cli(argv)
    # ROCm defaults (global constraints); explicit environment values win.
    os.environ.setdefault("MIOPEN_FIND_MODE", "2")
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

    models, dataloader, accelerator = _build_models_and_data(cli)
    compressor = models["compressor"]
    if not has_latent_stats(compressor):
        print(
            f"error: {type(compressor).__name__} has no latent_shift/latent_scale buffers "
            "(pre-v0.10.0 model); nothing to calibrate",
            file=sys.stderr,
        )
        sys.exit(2)

    max_batches = math.ceil(cli.num_images / cli.batch_size)
    autocast = accelerator.autocast if accelerator is not None else None
    print(f"Encoding up to {max_batches} batches of {cli.batch_size} (all resolutions)...")
    mean, std = calibrate_models(
        models, (imgs for imgs, _ in dataloader), max_batches, autocast=autocast
    )
    print("Raw per-channel latent stats (now stored as latent_shift / latent_scale):")
    print(format_latent_stats_table(mean, std, context_dims_of(compressor)))

    from fluxflow_training.training.checkpoint_manager import CheckpointManager

    out = Path(cli.out)
    CheckpointManager(output_dir=out.resolve().parent).save_models(
        diffuser=models["diffuser"],
        text_encoder=models["text_encoder"],
        model_config=models.get("model_config"),
        model_path=out,
    )
    print(f"✓ Calibrated checkpoint written to {out}")


if __name__ == "__main__":
    main()
