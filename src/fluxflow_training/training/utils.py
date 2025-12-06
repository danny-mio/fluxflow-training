"""Training utilities for FluxFlow (EMA, buffers, device detection)."""

import time
from typing import Any, Callable, Dict, Iterator, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FloatBuffer:
    """Rolling buffer for tracking scalar metrics (e.g., loss averages)."""

    def __init__(self, max_items: int):
        """
        Args:
            max_items: Maximum buffer size (oldest items dropped)
        """
        self.max_items = max_items
        self._items: list[float] = []

    def add_item(self, value: float) -> None:
        """Add a new value to the buffer (FIFO)."""
        self._items.append(float(value))
        if len(self._items) > self.max_items:
            self._items.pop(0)

    @property
    def average(self) -> float:
        """Compute average of buffered values."""
        if not self._items:
            return 0.0
        return float(sum(self._items) / len(self._items))


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: PyTorch model to track
            decay: EMA decay rate (higher = slower updates)
            device: Device to store shadow parameters (defaults to model device)
        """
        self.model = model
        self.decay = decay
        self.device = device or next(model.parameters()).device
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone().to(self.device) for k, v in model.state_dict().items()
        }
        self._original: Optional[Dict[str, torch.Tensor]] = None

    def update(self) -> None:
        """Update shadow parameters with current model state."""
        with torch.no_grad():
            for name, param in self.model.state_dict().items():
                if param.dtype.is_floating_point:
                    self.shadow[name].mul_(self.decay).add_(
                        param.data.to(self.device), alpha=1.0 - self.decay
                    )

    def apply_shadow(self) -> None:
        """Replace model parameters with EMA shadow (for inference/eval)."""
        self._original = {
            k: v.detach().clone().to(self.device) for k, v in self.model.state_dict().items()
        }
        self.model.load_state_dict(self.shadow)

    def restore(self) -> None:
        """Restore original model parameters (after apply_shadow)."""
        if self._original is not None:
            self.model.load_state_dict(self._original)
            self._original = None

    def copy_to(self, model: nn.Module) -> None:
        """
        Copy EMA shadow parameters to another model.

        Args:
            model: Target model to copy EMA parameters to
        """
        model.load_state_dict(self.shadow)

    def state_dict(self) -> Dict[str, Any]:
        """
        Return EMA state for checkpointing.

        Returns:
            Dictionary containing decay and shadow parameters
        """
        return {
            "decay": torch.tensor(self.decay),
            "shadow": self.shadow.copy(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load EMA state from checkpoint.

        Args:
            state_dict: Dictionary containing decay and shadow parameters
        """
        self.decay = float(state_dict["decay"])
        shadow_state = state_dict["shadow"]
        if isinstance(shadow_state, dict):
            self.shadow = shadow_state.copy()
        else:
            self.shadow = shadow_state
        # Ensure all shadow parameters are on the correct device
        self.shadow = {k: v.to(self.device) for k, v in self.shadow.items()}


def current_lr(optim: torch.optim.Optimizer) -> float:
    """
    Extract current learning rate from optimizer.

    Args:
        optim: PyTorch optimizer

    Returns:
        Learning rate of first param group
    """
    return float(optim.param_groups[0]["lr"])


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker initialization (placeholder for custom seeding).

    Args:
        worker_id: Worker process ID
    """


def img_to_random_packet(
    img: torch.Tensor,
    d_model: int = 128,
    downscales: int = 4,
    max_hw: int = 1024,
) -> torch.Tensor:
    """
    Generate random latent packet matching image dimensions.

    Used for training discriminator with random noise to improve
    VAE latent space regularization.

    Args:
        img: Input image tensor [B, C, H, W]
        d_model: Latent dimension
        downscales: Number of downsampling layers (default: 4)
        max_hw: Maximum spatial dimension for normalization

    Returns:
        Random packet [B, T+1, d_model] where T = (H//2^downscales) * (W//2^downscales)
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)
    B, _, H, W = img.shape
    H_lat = max(H // (2**downscales), 1)
    W_lat = max(W // (2**downscales), 1)
    T = H_lat * W_lat
    dtype = img.dtype if img.is_floating_point() else torch.float32
    tokens = torch.randn(B, T, d_model, device=img.device, dtype=dtype)
    hw = torch.zeros(B, 1, d_model, device=img.device, dtype=dtype)
    hw[:, 0, 0] = H_lat / float(max_hw)
    hw[:, 0, 1] = W_lat / float(max_hw)
    return torch.cat([tokens, hw], dim=1)


def get_device() -> torch.device:
    """
    Auto-detect best available device (CUDA > MPS > CPU).

    Returns:
        torch.device instance
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class RobustDataLoaderIterator:
    """
    Wrapper for DataLoader iteration with automatic error recovery and worker restart.

    Handles worker failures, timeouts, and connection errors by recreating the
    DataLoader and continuing iteration. Useful for streaming WebDatasets.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        dataloader_factory: Optional[Callable[[], DataLoader]] = None,
        max_retries: int = 5,
        retry_delay: float = 5.0,
        max_consecutive_errors: int = 10,
    ):
        """
        Args:
            dataloader: Initial DataLoader instance
            dataloader_factory: Optional factory function to recreate DataLoader on failure
                If None, will reuse the same DataLoader instance
            max_retries: Maximum number of retries per batch before raising
            retry_delay: Base delay between retries (with exponential backoff)
            max_consecutive_errors: Max consecutive batch errors before recreating DataLoader
        """
        self.dataloader = dataloader
        self.dataloader_factory = dataloader_factory
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_consecutive_errors = max_consecutive_errors

        self._iterator: Optional[Iterator] = None
        self._consecutive_errors = 0
        self._total_errors = 0
        self._batches_yielded = 0

    def __iter__(self) -> "RobustDataLoaderIterator":
        """Start iteration."""
        self._iterator = iter(self.dataloader)
        self._consecutive_errors = 0
        return self

    def __next__(self) -> Any:
        """Get next batch with error recovery."""
        if self._iterator is None:
            self._iterator = iter(self.dataloader)

        retry_count = 0
        last_error = None

        while retry_count <= self.max_retries:
            try:
                batch = next(self._iterator)
                self._consecutive_errors = 0
                self._batches_yielded += 1
                return batch

            except StopIteration:
                # Normal end of epoch
                raise

            except (RuntimeError, OSError, BrokenPipeError, ConnectionError) as e:
                last_error = e
                self._consecutive_errors += 1
                self._total_errors += 1

                error_msg = str(e).lower()
                is_worker_error = any(
                    x in error_msg
                    for x in [
                        "worker",
                        "dataloader",
                        "broken pipe",
                        "connection",
                        "timeout",
                        "eof",
                        "errno",
                        "reset by peer",
                    ]
                )

                if is_worker_error or self._consecutive_errors >= self.max_consecutive_errors:
                    # Recreate DataLoader to recover from worker failure
                    print(f"DataLoader error ({type(e).__name__}): {e}")
                    print(
                        f"Attempting to recreate DataLoader (errors: {self._consecutive_errors} consecutive, {self._total_errors} total)"
                    )

                    try:
                        self._recreate_dataloader()
                        self._consecutive_errors = 0
                        retry_count += 1
                        continue
                    except Exception as recreate_error:
                        print(f"Failed to recreate DataLoader: {recreate_error}")
                        retry_count += 1
                        delay = self.retry_delay * (2 ** (retry_count - 1))
                        print(f"Waiting {delay:.1f}s before retry {retry_count}/{self.max_retries}")
                        time.sleep(delay)
                        continue
                else:
                    # Try to continue with same iterator
                    retry_count += 1
                    delay = self.retry_delay * (2 ** (retry_count - 1))
                    print(
                        f"Batch error ({type(e).__name__}), retry {retry_count}/{self.max_retries} in {delay:.1f}s"
                    )
                    time.sleep(delay)

            except Exception as e:
                # Unexpected error - log and retry
                last_error = e  # type: ignore[assignment]
                self._consecutive_errors += 1
                self._total_errors += 1
                retry_count += 1

                if retry_count > self.max_retries:
                    break

                delay = self.retry_delay * (2 ** (retry_count - 1))
                print(
                    f"Unexpected error ({type(e).__name__}: {e}), retry {retry_count}/{self.max_retries} in {delay:.1f}s"
                )
                time.sleep(delay)

        # All retries exhausted
        print(f"Max retries ({self.max_retries}) exceeded after {self._total_errors} total errors")
        if last_error:
            raise last_error
        raise RuntimeError("DataLoader iteration failed after max retries")

    def _recreate_dataloader(self) -> None:
        """Recreate the DataLoader to recover from worker failures."""
        if self.dataloader_factory is not None:
            # Use factory to create fresh DataLoader
            old_dataloader = self.dataloader
            self.dataloader = self.dataloader_factory()

            # Try to clean up old dataloader
            try:
                if hasattr(old_dataloader, "_iterator"):
                    del old_dataloader._iterator
            except Exception:
                pass

        # Create new iterator
        self._iterator = iter(self.dataloader)
        print(
            f"DataLoader recreated successfully (yielded {self._batches_yielded} batches before failure)"
        )

    def get_stats(self) -> Dict[str, int]:
        """Get iteration statistics."""
        return {
            "batches_yielded": self._batches_yielded,
            "total_errors": self._total_errors,
            "consecutive_errors": self._consecutive_errors,
        }


def create_robust_dataloader_iterator(
    dataloader: DataLoader,
    dataloader_kwargs: Optional[Dict[str, Any]] = None,
    accelerator: Any = None,
    max_retries: int = 5,
    retry_delay: float = 5.0,
) -> RobustDataLoaderIterator:
    """
    Create a robust iterator for a DataLoader with automatic recovery.

    Args:
        dataloader: The DataLoader to wrap
        dataloader_kwargs: Optional kwargs to recreate DataLoader on failure
        accelerator: Optional Accelerator instance for preparing new DataLoaders
        max_retries: Maximum retries per batch
        retry_delay: Base delay between retries

    Returns:
        RobustDataLoaderIterator instance
    """

    def create_factory() -> Optional[Callable[[], DataLoader]]:
        if dataloader_kwargs is None:
            return None

        kwargs = dataloader_kwargs  # Capture for closure

        def factory() -> DataLoader:
            new_loader = DataLoader(**kwargs)
            if accelerator is not None:
                new_loader = accelerator.prepare(new_loader)
            return new_loader

        return factory

    return RobustDataLoaderIterator(
        dataloader=dataloader,
        dataloader_factory=create_factory(),
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
