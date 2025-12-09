"""Checkpoint management for FluxFlow training.

Handles saving and loading of model checkpoints, optimizer states,
scheduler states, EMA states, and training state metadata.
"""

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import safetensors.torch
import torch
from fluxflow.utils import get_logger
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler  # type: ignore[attr-defined]

logger = get_logger(__name__)


def _move_to_cpu(obj):
    """Recursively move all tensors in a nested structure to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_move_to_cpu(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_move_to_cpu(item) for item in obj)
    else:
        return obj


class CheckpointManager:
    """
    Manages all checkpoint save/load operations for training.

    Handles:
    - Model checkpoints (parallel loading)
    - Optimizer states
    - Scheduler states
    - EMA states
    - Training state metadata (epoch, batch, global_step, etc.)
    - Sampler states (for mid-epoch resume)
    - Learning rate persistence

    Example:
        >>> manager = CheckpointManager(output_dir="outputs/")
        >>> manager.save_models(diffuser, text_encoder)
        >>> manager.save_training_state(
        ...     epoch=5, batch_idx=100, global_step=5000,
        ...     optimizers={"flow": opt_flow}, schedulers={"flow": sched_flow}
        ... )
    """

    def __init__(self, output_dir: str | Path, generate_diagrams: bool = False):
        """
        Initialize checkpoint manager.

        Args:
            output_dir: Directory to save/load checkpoints
            generate_diagrams: Whether to automatically generate training diagrams on checkpoint save
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generate_diagrams = generate_diagrams

        # File paths
        self.model_path = self.output_dir / "flxflow_final.safetensors"
        self.text_encoder_path = self.output_dir / "text_encoder.safetensors"
        self.image_encoder_path = self.output_dir / "image_encoder.safetensors"
        self.training_state_path = self.output_dir / "training_state.json"
        self.training_states_path = self.output_dir / "training_states.pt"
        self.sampler_state_path = self.output_dir / "sampler_state.pt"
        self.lr_save_path = self.output_dir / "lr_sav.json"

    def save_models(
        self,
        diffuser: nn.Module,
        text_encoder: nn.Module,
        discriminators: Optional[dict[str, nn.Module]] = None,
        save_pretrained: bool = False,
    ) -> None:
        """
        Save model checkpoints.

        Args:
            diffuser: Main diffusion model (contains compressor, flow, expander)
            text_encoder: Text encoder model
            discriminators: Optional dict of discriminator models
            save_pretrained: Whether to save in HuggingFace format
        """
        logger.info(f"Saving model checkpoints to {self.output_dir}")

        # Save diffuser (compressor + flow + expander)
        diffuser_state = {f"diffuser.{k}": v for k, v in diffuser.state_dict().items()}
        safetensors.torch.save_file(diffuser_state, str(self.model_path))
        logger.debug(f"✓ Saved diffuser to {self.model_path}")

        # Save text encoder
        text_encoder_state = {f"text_encoder.{k}": v for k, v in text_encoder.state_dict().items()}
        safetensors.torch.save_file(text_encoder_state, str(self.text_encoder_path))
        logger.debug(f"✓ Saved text encoder to {self.text_encoder_path}")

        # Save discriminators if provided
        if discriminators:
            for name, disc in discriminators.items():
                disc_path = self.output_dir / f"{name}.safetensors"
                disc_state = {f"{name}.{k}": v for k, v in disc.state_dict().items()}
                safetensors.torch.save_file(disc_state, str(disc_path))
                logger.debug(f"✓ Saved {name} to {disc_path}")

    def load_models_parallel(
        self, checkpoint_path: Optional[Union[str, Path]] = None
    ) -> dict[str, Optional[dict[str, Any]]]:
        """
        Load model checkpoints in parallel for faster resume.

        Args:
            checkpoint_path: Optional path to main checkpoint (uses default if None)

        Returns:
            Dictionary mapping component names to loaded state dicts
        """
        if checkpoint_path is None:
            checkpoint_path = self.model_path
        else:
            checkpoint_path = Path(checkpoint_path)

        # Build list of files to load
        files_to_load = []

        if self.text_encoder_path.exists():
            files_to_load.append((str(self.text_encoder_path), "text_encoder"))
        if self.image_encoder_path.exists():
            files_to_load.append((str(self.image_encoder_path), "image_encoder"))

        # Load discriminators if they exist
        d_img_path = self.output_dir / "D_img.safetensors"
        if d_img_path.exists():
            files_to_load.append((str(d_img_path), "D_img"))

        if checkpoint_path.exists():
            # Load main checkpoint components
            files_to_load.append((str(checkpoint_path), "diffuser.compressor"))
            files_to_load.append((str(checkpoint_path), "diffuser.flow_processor"))
            files_to_load.append((str(checkpoint_path), "diffuser.expander"))

            # Load text encoder from main checkpoint if separate file doesn't exist
            if not self.text_encoder_path.exists():
                files_to_load.append((str(checkpoint_path), "text_encoder"))

        if not files_to_load:
            logger.warning("No model checkpoints found")
            return {}

        logger.info(f"Loading {len(files_to_load)} checkpoint components in parallel...")

        def load_single(path_prefix):
            path, prefix = path_prefix
            if not os.path.exists(path):
                return prefix, None
            state_dict = safetensors.torch.load_file(path)
            # Filter keys with the given prefix
            filtered = {
                k.replace(f"{prefix}.", ""): v
                for k, v in state_dict.items()
                if k.startswith(f"{prefix}.")
            }
            return prefix, filtered if filtered else None

        with ThreadPoolExecutor(max_workers=min(len(files_to_load), 4)) as executor:
            results = dict(executor.map(load_single, files_to_load))

        # Log what was loaded
        for component, state in results.items():
            if state:
                logger.debug(f"✓ Loaded {component}")

        return results

    def save_training_state(
        self,
        epoch: int,
        batch_idx: int,
        global_step: int,
        optimizers: dict[str, Optimizer],
        schedulers: dict[str, _LRScheduler],  # type: ignore[type-arg]
        ema: Any,
        sampler: Any = None,
        kl_beta_current: float = 0.0,
        kl_warmup_steps: int = 0,
        kl_max_beta: float = 0.0,
        pipeline_metadata: Optional[dict] = None,
    ) -> None:
        """
        Save complete training state for resume.

        Args:
            epoch: Current epoch number
            batch_idx: Current batch index
            global_step: Global training step counter
            optimizers: Dictionary of optimizers to save
            schedulers: Dictionary of schedulers to save
            ema: EMA object with state_dict method
            sampler: Optional sampler with state_dict method
            kl_beta_current: Current KL divergence beta value
            kl_warmup_steps: Total KL warmup steps
            kl_max_beta: Maximum KL beta value
            pipeline_metadata: Optional pipeline metadata (for multi-step training)
        """
        logger.info(f"Saving training state at epoch {epoch}, batch {batch_idx}")

        # Save learning rates
        lr_dict = {name: opt.param_groups[0]["lr"] for name, opt in optimizers.items()}
        with open(self.lr_save_path, "w") as f:
            json.dump(lr_dict, f)

        # Determine version and mode
        version = "2.0" if pipeline_metadata else "1.0"
        mode = "pipeline" if pipeline_metadata else "legacy"

        # Save training state metadata
        training_state = {
            "version": version,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "batch_idx": batch_idx,
            "global_step": global_step,
            "kl_warmup": {
                "current_beta": kl_beta_current,
                "warmup_steps": kl_warmup_steps,
                "max_beta": kl_max_beta,
                "progress_pct": (
                    min(100.0 * global_step / kl_warmup_steps, 100.0)
                    if kl_warmup_steps > 0
                    else 100.0
                ),
            },
            "learning_rates": lr_dict,
        }

        # Add pipeline metadata if provided
        if pipeline_metadata:
            training_state["pipeline"] = pipeline_metadata

        with open(self.training_state_path, "w") as f:
            json.dump(training_state, f, indent=2)

        # Save consolidated training states (optimizer, scheduler, EMA)
        # Create backup of previous checkpoint
        if self.training_states_path.exists():
            backup_path = self.training_states_path.with_suffix(".pt.bck")
            if backup_path.exists():
                backup_path.unlink()
            shutil.move(self.training_states_path, backup_path)

        consolidated_state = {
            "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
            "schedulers": {name: sched.state_dict() for name, sched in schedulers.items()},
        }

        # Only save EMA if it exists (not all steps use EMA)
        if ema is not None:
            consolidated_state["ema"] = ema.state_dict()
        # Move all tensors to CPU before saving to avoid CUDA issues
        consolidated_state = _move_to_cpu(consolidated_state)
        torch.save(consolidated_state, self.training_states_path)

        # Save sampler state for mid-epoch resume
        if sampler is not None and hasattr(sampler, "state_dict"):
            backup_path = self.sampler_state_path.with_suffix(".pt.bck")
            if self.sampler_state_path.exists():
                if backup_path.exists():
                    backup_path.unlink()
                shutil.move(self.sampler_state_path, backup_path)
            sampler_state = _move_to_cpu(sampler.state_dict())
            torch.save(sampler_state, self.sampler_state_path)

        logger.debug(f"✓ Training state saved")

        # Generate training diagrams if enabled
        if self.generate_diagrams:
            try:
                from fluxflow_training.scripts.generate_training_graphs import generate_all_diagrams

                logger.debug("Generating training diagrams...")
                success = generate_all_diagrams(self.output_dir, verbose=False)
                if success:
                    logger.debug("✓ Training diagrams generated")
                else:
                    logger.warning("Failed to generate training diagrams")
            except Exception as e:
                logger.warning(f"Error generating diagrams: {e}")

    def load_training_state(self) -> Optional[dict]:
        """
        Load training state metadata for resume.

        Returns:
            Dictionary with epoch, batch_idx, global_step, etc. or None if not found
        """
        if not self.training_state_path.exists():
            logger.debug("No training state found")
            return None

        try:
            with open(self.training_state_path, "r") as f:
                state = json.load(f)
            logger.info(
                f"Loaded training state: epoch {state.get('epoch', 0)}, "
                f"batch {state.get('batch_idx', 0)}, "
                f"global step {state.get('global_step', 0)}"
            )
            return state
        except Exception as e:
            logger.warning(f"Could not load training state: {e}")
            return None

    def load_optimizer_scheduler_ema_states(
        self,
        optimizers: dict[str, Optimizer],
        schedulers: dict[str, _LRScheduler],  # type: ignore[type-arg]
        ema: Any,
    ) -> bool:
        """
        Load optimizer, scheduler, and EMA states.

        Args:
            optimizers: Dictionary of optimizers to restore
            schedulers: Dictionary of schedulers to restore
            ema: EMA object to restore

        Returns:
            True if states were loaded, False otherwise
        """
        # Try consolidated checkpoint first (new format)
        if self.training_states_path.exists():
            logger.info(f"Loading consolidated training states from {self.training_states_path}")
            try:
                training_states = torch.load(
                    self.training_states_path,
                    weights_only=False,
                    map_location="cpu",
                    mmap=True,
                )

                if "optimizers" in training_states:
                    for name, opt in optimizers.items():
                        if name in training_states["optimizers"]:
                            opt.load_state_dict(training_states["optimizers"][name])
                    logger.debug("✓ Optimizer states restored")

                if "schedulers" in training_states:
                    for name, sched in schedulers.items():
                        if name in training_states["schedulers"]:
                            sched.load_state_dict(training_states["schedulers"][name])  # type: ignore[arg-type]
                    logger.debug("✓ Scheduler states restored")

                if "ema" in training_states:
                    ema.load_state_dict(training_states["ema"])
                    logger.debug("✓ EMA state restored")

                return True

            except Exception as e:
                logger.warning(f"Could not load consolidated training states: {e}")
                return False

        logger.debug("No consolidated training states found")
        return False

    def load_sampler_state(self) -> Optional[dict]:
        """
        Load sampler state for mid-epoch resume.

        Returns:
            Sampler state dict or None if not found
        """
        if not self.sampler_state_path.exists():
            return None

        try:
            state = torch.load(self.sampler_state_path, weights_only=False, mmap=True)
            logger.info(
                f"Loaded sampler state: epoch {state.get('current_epoch', 0)}, "
                f"position {state.get('position', 0)}"
            )
            return state
        except Exception as e:
            logger.warning(f"Could not load sampler state: {e}")
            return None

    def load_learning_rates(self) -> Optional[dict[str, float]]:
        """
        Load saved learning rates.

        Returns:
            Dictionary mapping optimizer names to learning rates, or None
        """
        if not self.lr_save_path.exists():
            return None

        try:
            with open(self.lr_save_path, "r") as f:
                lr_dict = json.load(f)
            logger.info(f"Loaded learning rates: {lr_dict}")
            return lr_dict
        except Exception as e:
            logger.warning(f"Could not load learning rates: {e}")
            return None
