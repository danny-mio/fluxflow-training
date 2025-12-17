"""
Training Progress Logger

Logs training metrics to a structured format for visualization and analysis.
Creates a 'graph' folder in the output path and maintains session continuity
across training interruptions.
"""

import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize metrics to handle NaN/Inf values.

    Args:
        metrics: Dictionary of metrics

    Returns:
        Sanitized metrics dictionary with NaN/Inf replaced by None
    """
    sanitized = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                sanitized[key] = None
            else:
                sanitized[key] = value
        else:
            sanitized[key] = value
    return sanitized


class TrainingProgressLogger:
    """
    Logs training progress metrics to disk for later visualization.

    Features:
    - Creates 'graph' folder in output path
    - Maintains session continuity across restarts
    - Logs metrics in JSON Lines format for easy parsing
    - Tracks session metadata
    """

    def __init__(
        self,
        output_path: str,
        step_name: Optional[str] = None,
        flush_policy: str = "immediate",
        flush_interval: int = 50,
    ):
        """
        Initialize the progress logger.

        Args:
            output_path: Base output path for training artifacts
            step_name: Optional pipeline step name for file naming
            flush_policy: Flush policy ("immediate", "periodic", "buffered")
            flush_interval: Number of writes before flush (for periodic policy)
        """
        self.output_path = Path(output_path)
        self.graph_dir = self.output_path / "graph"
        self.graph_dir.mkdir(parents=True, exist_ok=True)

        # Track current step for pipeline mode
        self.current_step_name = step_name

        # Flush policy configuration
        self.flush_policy = flush_policy
        self.flush_interval = flush_interval
        self._write_count = 0

        # File paths (will be updated if step_name changes)
        self._update_file_paths()

        # Initialize or load session
        self.session_id = self._get_or_create_session()
        self.session_start_time = time.time()

        # Track if this is a new session or resumed
        self.is_resumed = self._check_if_resumed()

    def _update_file_paths(self):
        """Update file paths based on current step name."""
        if self.current_step_name:
            # Step-specific files: session_step1.json, training_metrics_step1.jsonl
            self.session_file = self.graph_dir / f"session_{self.current_step_name}.json"
            self.metrics_file = self.graph_dir / f"training_metrics_{self.current_step_name}.jsonl"
        else:
            # Default files (legacy/non-pipeline mode)
            self.session_file = self.graph_dir / "session.json"
            self.metrics_file = self.graph_dir / "training_metrics.jsonl"

    def set_step(self, step_name: str):
        """
        Set the current pipeline step name and update file paths.

        Args:
            step_name: Pipeline step name (e.g., "gan_warmup", "vae_gan_lpips")
        """
        self.current_step_name = step_name
        self._update_file_paths()

        # Reload session for new step
        self.session_id = self._get_or_create_session()
        self.is_resumed = self._check_if_resumed()

    def _get_or_create_session(self) -> str:
        """
        Get existing session ID or create a new one.

        Session ID is based on the initial training start time to ensure
        continuity across interruptions and restarts.

        Returns:
            Session ID string
        """
        if self.session_file.exists():
            try:
                with open(self.session_file, "r") as f:
                    session_data = json.load(f)
                    session_id = session_data.get("session_id", self._create_new_session_id())
                    return str(session_id)
            except (json.JSONDecodeError, IOError):
                # If session file is corrupted, create new session
                return self._create_new_session_id()
        else:
            session_id = self._create_new_session_id()
            self._save_session_metadata(session_id, is_new=True)
            return session_id

    def _create_new_session_id(self) -> str:
        """Create a new session ID based on timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _check_if_resumed(self) -> bool:
        """Check if this is a resumed training session."""
        if self.session_file.exists():
            try:
                with open(self.session_file, "r") as f:
                    session_data = json.load(f)
                    resumed_count = session_data.get("resumed_count", 0)
                    return bool(resumed_count > 0)
            except (json.JSONDecodeError, IOError):
                return False
        return False

    def _save_session_metadata(self, session_id: str, is_new: bool = False):
        """
        Save session metadata to disk.

        Args:
            session_id: The session identifier
            is_new: Whether this is a brand new session
        """
        session_data = {}

        # Load existing data if available
        if self.session_file.exists() and not is_new:
            try:
                with open(self.session_file, "r") as f:
                    session_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Update session data
        if is_new:
            session_data = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "resumed_count": 0,
                "last_update": datetime.now().isoformat(),
            }
        else:
            session_data["resumed_count"] = session_data.get("resumed_count", 0) + 1
            session_data["last_update"] = datetime.now().isoformat()
            session_data["last_resume_at"] = datetime.now().isoformat()

        # Save to disk
        with open(self.session_file, "w") as f:
            json.dump(session_data, f, indent=2)

    def log_metrics(
        self,
        epoch: int,
        batch: int,
        global_step: int,
        metrics: Dict[str, float],
        learning_rates: Optional[Dict[str, float]] = None,
        extras: Optional[Dict[str, Any]] = None,
    ):
        """
        Log training metrics to disk.

        Args:
            epoch: Current epoch number
            batch: Current batch number within epoch
            global_step: Global training step counter
            metrics: Dictionary of metric names to values (e.g., {'vae_loss': 0.123, 'kl_loss': 2.5})
            learning_rates: Optional dictionary of learning rates (e.g., {'lr': 1e-4, 'vae': 5e-5})
            extras: Optional dictionary of additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "epoch": epoch,
            "batch": batch,
            "global_step": global_step,
            "metrics": _sanitize_metrics(metrics),
        }

        if learning_rates:
            log_entry["learning_rates"] = learning_rates

        if extras:
            log_entry["extras"] = extras

        try:
            json_str = json.dumps(log_entry, allow_nan=False)
            with open(self.metrics_file, "a") as f:
                f.write(json_str + "\n")
                self._apply_flush_policy(f)
        except (IOError, OSError) as e:
            logger.error(f"Failed to write metrics: {e}")
            try:
                print(f"FALLBACK_METRIC: {json_str}", file=sys.stderr)
            except Exception:
                pass
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to serialize metrics: {e}")

    def _apply_flush_policy(self, f):
        """Apply the configured flush policy to file handle.

        Args:
            f: Open file handle
        """
        if self.flush_policy == "immediate":
            f.flush()
        elif self.flush_policy == "periodic":
            self._write_count += 1
            if self._write_count >= self.flush_interval:
                f.flush()
                self._write_count = 0
        # "buffered" does nothing - relies on OS buffering

    def mark_resumed(self):
        """Mark this session as resumed (called when training is restarted)."""
        self._save_session_metadata(self.session_id, is_new=False)

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get current session information.

        Returns:
            Dictionary containing session metadata
        """
        if self.session_file.exists():
            try:
                with open(self.session_file, "r") as f:
                    data = json.load(f)
                    return dict(data) if isinstance(data, dict) else {}
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def get_metrics_file_path(self) -> Path:
        """Get the path to the metrics file."""
        return self.metrics_file

    def get_graph_dir(self) -> Path:
        """Get the path to the graph directory."""
        return self.graph_dir
