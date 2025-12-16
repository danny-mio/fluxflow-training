"""FluxFlow training utilities (losses, schedulers, EMA, buffers)."""

from .checkpoint_manager import CheckpointManager
from .flow_trainer import FlowTrainer
from .losses import (
    compute_mmd,
    d_hinge_loss,
    g_hinge_loss,
    kl_standard_normal,
    r1_penalty,
)
from .pipeline_config import (
    OptimizationConfig,
    OptimizerConfig,
    PipelineConfig,
    PipelineConfigValidator,
    PipelineStepConfig,
    SchedulerConfig,
    TransitionCriteria,
    parse_pipeline_config,
)
from .pipeline_orchestrator import TrainingPipelineOrchestrator
from .progress_logger import TrainingProgressLogger
from .schedulers import (
    cosine_anneal_beta,
    sample_t,
)
from .training_mode import TrainingComponent, TrainingMode
from .utils import (
    EMA,
    FloatBuffer,
    RobustDataLoaderIterator,
    create_robust_dataloader_iterator,
    current_lr,
    get_device,
    img_to_random_packet,
    worker_init_fn,
)
from .vae_trainer import VAETrainer

__all__ = [
    # Losses
    "d_hinge_loss",
    "g_hinge_loss",
    "r1_penalty",
    "kl_standard_normal",
    "compute_mmd",
    # Schedulers
    "cosine_anneal_beta",
    "sample_t",
    # Training utils
    "FloatBuffer",
    "EMA",
    "current_lr",
    "worker_init_fn",
    "get_device",
    "img_to_random_packet",
    "TrainingProgressLogger",
    "RobustDataLoaderIterator",
    "create_robust_dataloader_iterator",
    # Checkpoint management
    "CheckpointManager",
    # Trainers
    "VAETrainer",
    "FlowTrainer",
    # Training modes
    "TrainingComponent",
    "TrainingMode",
    # Pipeline configuration
    "PipelineConfig",
    "PipelineStepConfig",
    "PipelineConfigValidator",
    "TransitionCriteria",
    "OptimizerConfig",
    "SchedulerConfig",
    "OptimizationConfig",
    "parse_pipeline_config",
    # Pipeline orchestration
    "TrainingPipelineOrchestrator",
]
