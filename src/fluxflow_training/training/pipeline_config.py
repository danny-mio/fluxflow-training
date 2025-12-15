"""Pipeline configuration and validation for multi-step training.

This module provides dataclasses and validation logic for defining and validating
multi-step training pipelines with configurable freeze/unfreeze, optimizer configs,
and transition criteria.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from fluxflow.utils import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a single dataset (local or webdataset)."""

    # Dataset type
    type: Literal["local", "webdataset"] = "local"

    # Local dataset configuration
    image_folder: Optional[str] = None
    captions_file: Optional[str] = None

    # WebDataset configuration
    webdataset_url: Optional[str] = None
    webdataset_token: Optional[str] = None
    webdataset_image_key: str = "png"
    webdataset_label_key: str = "json"
    webdataset_caption_key: str = "prompt"
    webdataset_size: int = 10000
    webdataset_samples_per_shard: int = 1000

    # Common configuration
    batch_size: Optional[int] = None  # Override step batch_size
    workers: Optional[int] = None  # Override step workers


@dataclass
class TransitionCriteria:
    """Defines when to transition to the next pipeline step."""

    mode: Literal["epoch", "loss_threshold"] = "epoch"
    value: Optional[int] = None  # For epoch mode: number of epochs
    metric: Optional[str] = None  # For loss_threshold mode: metric name
    threshold: Optional[float] = None  # For loss_threshold mode: threshold value
    max_epochs: Optional[int] = None  # For loss_threshold mode: upper limit


@dataclass
class OptimizerConfig:
    """Optimizer configuration for a single model component."""

    type: str = "AdamW"
    lr: float = 1e-5
    betas: Optional[tuple[float, float]] = None
    momentum: Optional[float] = None
    weight_decay: float = 0.0
    eps: float = 1e-8
    alpha: Optional[float] = None  # For RMSprop
    centered: bool = False  # For RMSprop
    amsgrad: bool = False  # For AdamW
    decoupled_weight_decay: bool = True  # For Lion


@dataclass
class SchedulerConfig:
    """Scheduler configuration for a single model component."""

    type: str = "CosineAnnealingLR"
    eta_min_factor: float = 0.1  # For CosineAnnealingLR
    start_factor: float = 1.0  # For LinearLR
    end_factor: float = 0.1  # For LinearLR
    total_iters: Optional[int] = None  # For LinearLR
    step_size: Optional[int] = None  # For StepLR
    gamma: float = 0.1  # For StepLR/ExponentialLR
    mode: str = "min"  # For ReduceLROnPlateau
    factor: float = 0.1  # For ReduceLROnPlateau
    patience: int = 10  # For ReduceLROnPlateau


@dataclass
class OptimizationConfig:
    """Combined optimizer and scheduler configuration."""

    optimizers: dict[str, OptimizerConfig] = field(default_factory=dict)
    schedulers: dict[str, SchedulerConfig] = field(default_factory=dict)


@dataclass
class PipelineStepConfig:
    """Configuration for a single pipeline step."""

    # Required fields
    name: str
    n_epochs: int

    # Optional descriptive fields
    description: str = ""

    # Dataset selection (optional - uses default if not specified)
    dataset: Optional[str] = None  # Name of dataset from datasets dict

    # Training modes
    train_vae: bool = False
    gan_training: bool = False
    use_lpips: bool = False
    train_spade: bool = False
    train_diff: bool = False
    train_diff_full: bool = False
    use_ema: bool = True  # Exponential Moving Average (costs 2x model VRAM)

    # Classifier-Free Guidance (CFG) for text-conditioned flow training
    cfg_dropout_prob: float = 0.0  # Probability of null conditioning (0.0=disabled, 0.10=standard)

    # Model freeze/unfreeze
    freeze: list[str] = field(default_factory=list)
    unfreeze: list[str] = field(default_factory=list)

    # Training hyperparameters
    batch_size: int = 2
    workers: int = 8
    lr: Optional[float] = None
    lr_min: float = 0.1
    use_fp16: bool = False
    initial_clipping_norm: float = 1.0
    max_steps: Optional[int] = None  # Limit batches per epoch (for testing)

    # Loss weights
    kl_beta: float = 0.0001
    kl_warmup_steps: int = 5000
    kl_free_bits: float = 0.0
    lambda_adv: float = 0.5
    lambda_lpips: float = 0.1
    mse_weight: float = 0.1

    # GAN settings
    r1_interval: int = 16
    r1_gamma: float = 5.0
    instance_noise_std: float = 0.01
    instance_noise_decay: float = 0.9999
    adaptive_weights: bool = True

    # Optimization configuration (inline YAML)
    optimization: Optional[OptimizationConfig] = None

    # Transition criteria
    transition_on: TransitionCriteria = field(default_factory=TransitionCriteria)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    steps: list[PipelineStepConfig]
    defaults: Optional[PipelineStepConfig] = None
    datasets: dict[str, DatasetConfig] = field(default_factory=dict)
    default_dataset: Optional[str] = None  # Name of default dataset


class PipelineConfigValidator:
    """Validates pipeline configurations before training starts."""

    VALID_COMPONENTS = {
        "compressor",
        "expander",
        "flow_processor",
        "flow",
        "text_encoder",
        "discriminator",
        "D_img",
    }

    VALID_OPTIMIZER_TYPES = {
        "Adam",
        "AdamW",
        "SGD",
        "RMSprop",
        "Adagrad",
        "Adadelta",
        "Lion",
    }

    VALID_SCHEDULER_TYPES = {
        "CosineAnnealingLR",
        "LinearLR",
        "StepLR",
        "ExponentialLR",
        "ReduceLROnPlateau",
        "MultiStepLR",
        "ConstantLR",
    }

    VALID_METRICS = {
        "vae",
        "vae_loss",
        "recon",
        "recon_loss",
        "kl",
        "kl_loss",
        "discriminator",
        "generator",
        "lpips",
        "flow",
        "flow_loss",
    }

    def __init__(self, config: PipelineConfig):
        """Initialize validator with pipeline config."""
        self.config = config
        self.errors: list[str] = []

    def validate(self) -> list[str]:
        """
        Validate entire pipeline configuration.

        Returns:
            List of error messages (empty if valid)
        """
        self.errors = []

        if not self.config.steps:
            self.errors.append("Pipeline has no steps defined")
            return self.errors

        # Validate datasets
        self._validate_datasets()

        # Validate steps
        for i, step in enumerate(self.config.steps):
            step_name = step.name or f"step_{i}"
            self._validate_step(step, step_name, i)

        return self.errors

    def _validate_datasets(self) -> None:
        """Validate dataset configurations."""
        # Check if default_dataset exists in datasets dict
        if self.config.default_dataset:
            if not self.config.datasets:
                self.errors.append(
                    f"default_dataset '{self.config.default_dataset}' specified "
                    f"but no datasets configured"
                )
            elif self.config.default_dataset not in self.config.datasets:
                self.errors.append(
                    f"default_dataset '{self.config.default_dataset}' not found in datasets. "
                    f"Available: {', '.join(self.config.datasets.keys())}"
                )

        # Validate each dataset configuration
        for name, dataset in self.config.datasets.items():
            if dataset.type == "local":
                # Local dataset requires image_folder and captions_file
                if not dataset.image_folder:
                    self.errors.append(f"Dataset '{name}': type='local' requires 'image_folder'")
                if not dataset.captions_file:
                    self.errors.append(f"Dataset '{name}': type='local' requires 'captions_file'")
            elif dataset.type == "webdataset":
                # WebDataset requires webdataset_url and webdataset_token
                if not dataset.webdataset_url:
                    self.errors.append(
                        f"Dataset '{name}': type='webdataset' requires 'webdataset_url'"
                    )
                if not dataset.webdataset_token:
                    self.errors.append(
                        f"Dataset '{name}': type='webdataset' requires 'webdataset_token'"
                    )
            else:
                self.errors.append(
                    f"Dataset '{name}': unknown type '{dataset.type}'. "
                    f"Valid types: 'local', 'webdataset'"
                )

    def _validate_step(self, step: PipelineStepConfig, step_name: str, step_index: int) -> None:
        """Validate a single pipeline step."""
        # Check epoch count
        if step.n_epochs <= 0:
            self.errors.append(f"Step '{step_name}' (step {step_index + 1}): n_epochs must be > 0")

        # Validate dataset reference
        if step.dataset:
            if not self.config.datasets:
                self.errors.append(
                    f"Step '{step_name}' (step {step_index + 1}): "
                    f"references dataset '{step.dataset}' but no datasets configured"
                )
            elif step.dataset not in self.config.datasets:
                self.errors.append(
                    f"Step '{step_name}' (step {step_index + 1}): "
                    f"references unknown dataset '{step.dataset}'. "
                    f"Available: {', '.join(self.config.datasets.keys())}"
                )

        # Check training modes
        training_modes = {
            "train_vae": step.train_vae,
            "gan_training": step.gan_training,
            "train_spade": step.train_spade,
            "train_diff": step.train_diff,
            "train_diff_full": step.train_diff_full,
        }

        if not any(training_modes.values()):
            self.errors.append(
                f"Step '{step_name}' (step {step_index + 1}): "
                f"At least one training mode must be enabled"
            )

        # Validate max_steps
        if step.max_steps is not None and step.max_steps <= 0:
            self.errors.append(
                f"Step '{step_name}' (step {step_index + 1}): max_steps must be > 0 if specified"
            )

        # Validate freeze/unfreeze
        self._validate_freeze_unfreeze(step, step_name, step_index, training_modes)

        # Validate optimization config
        if step.optimization:
            self._validate_optimization(step.optimization, step_name, step_index)

        # Validate transition criteria
        self._validate_transition(step.transition_on, step_name, step_index)

    def _validate_freeze_unfreeze(
        self,
        step: PipelineStepConfig,
        step_name: str,
        step_index: int,
        training_modes: dict[str, bool],
    ) -> None:
        """Validate freeze/unfreeze configuration."""
        freeze_set = set(step.freeze)
        unfreeze_set = set(step.unfreeze)

        # Check for invalid component names
        for component in freeze_set | unfreeze_set:
            if component not in self.VALID_COMPONENTS:
                self.errors.append(
                    f"Step '{step_name}' (step {step_index + 1}): "
                    f"Unknown component '{component}'. "
                    f"Valid components: {', '.join(sorted(self.VALID_COMPONENTS))}"
                )

        # Determine which components should be trainable
        trainable_components = self._get_trainable_components(training_modes)

        # Check for conflicts (frozen but being trained)
        conflicts = freeze_set & trainable_components
        if conflicts:
            self.errors.append(
                f"Step '{step_name}' (step {step_index + 1}): "
                f"Cannot freeze and train the same component: {', '.join(sorted(conflicts))}. "
                f"Remove from freeze list or disable corresponding training mode."
            )

    def _get_trainable_components(self, training_modes: dict[str, bool]) -> set[str]:
        """Determine which components are being trained based on training modes."""
        trainable = set()

        if training_modes["train_vae"]:
            trainable.update(["compressor", "expander"])

        if training_modes["train_diff"] or training_modes["train_diff_full"]:
            trainable.update(["flow_processor", "flow"])

        if training_modes["gan_training"]:
            trainable.update(["discriminator", "D_img"])

        # Note: text_encoder is NOT automatically trained unless explicitly enabled
        # Users can freeze it without conflict

        return trainable

    def _validate_optimization(
        self, opt_config: OptimizationConfig, step_name: str, step_index: int
    ) -> None:
        """Validate optimization configuration."""
        # Validate optimizer types
        for name, opt in opt_config.optimizers.items():
            if opt.type not in self.VALID_OPTIMIZER_TYPES:
                self.errors.append(
                    f"Step '{step_name}' (step {step_index + 1}): "
                    f"Unknown optimizer type '{opt.type}' for '{name}'. "
                    f"Valid types: {', '.join(sorted(self.VALID_OPTIMIZER_TYPES))}"
                )

            # Validate optimizer-specific parameters (betas has default value, no validation needed)

        # Validate scheduler types
        for name, sched in opt_config.schedulers.items():
            if sched.type not in self.VALID_SCHEDULER_TYPES:
                self.errors.append(
                    f"Step '{step_name}' (step {step_index + 1}): "
                    f"Unknown scheduler type '{sched.type}' for '{name}'. "
                    f"Valid types: {', '.join(sorted(self.VALID_SCHEDULER_TYPES))}"
                )

    def _validate_transition(
        self, criteria: TransitionCriteria, step_name: str, step_index: int
    ) -> None:
        """Validate transition criteria."""
        if criteria.mode == "epoch":
            # Epoch mode transitions use n_epochs from the step, no separate value needed
            pass

        elif criteria.mode == "loss_threshold":
            if criteria.metric is None:
                self.errors.append(
                    f"Step '{step_name}' (step {step_index + 1}): "
                    f"Loss-threshold transition requires 'metric' field"
                )
            elif criteria.metric not in self.VALID_METRICS:
                self.errors.append(
                    f"Step '{step_name}' (step {step_index + 1}): "
                    f"Unknown metric '{criteria.metric}'. "
                    f"Valid metrics: {', '.join(sorted(self.VALID_METRICS))}"
                )

            if criteria.threshold is None:
                self.errors.append(
                    f"Step '{step_name}' (step {step_index + 1}): "
                    f"Loss-threshold transition requires 'threshold' field"
                )

            if criteria.max_epochs is None or criteria.max_epochs <= 0:
                self.errors.append(
                    f"Step '{step_name}' (step {step_index + 1}): "
                    f"Loss-threshold transition requires 'max_epochs' > 0 as safety limit"
                )


def _parse_dataset_config(dataset_dict: dict) -> DatasetConfig:
    """Parse a single dataset configuration."""
    return DatasetConfig(
        type=dataset_dict.get("type", "local"),
        # Local dataset fields
        image_folder=dataset_dict.get("image_folder"),
        captions_file=dataset_dict.get("captions_file"),
        # WebDataset fields
        webdataset_url=dataset_dict.get("webdataset_url"),
        webdataset_token=dataset_dict.get("webdataset_token"),
        webdataset_image_key=dataset_dict.get("webdataset_image_key", "png"),
        webdataset_label_key=dataset_dict.get("webdataset_label_key", "json"),
        webdataset_caption_key=dataset_dict.get("webdataset_caption_key", "prompt"),
        webdataset_size=dataset_dict.get("webdataset_size", 10000),
        webdataset_samples_per_shard=dataset_dict.get("webdataset_samples_per_shard", 1000),
        # Common fields
        batch_size=dataset_dict.get("batch_size"),
        workers=dataset_dict.get("workers"),
    )


def parse_pipeline_config(config_dict: dict) -> PipelineConfig:
    """
    Parse pipeline configuration from dictionary.

    Args:
        config_dict: Dictionary from YAML config (training.pipeline section)

    Returns:
        PipelineConfig instance

    Raises:
        ValueError: If pipeline configuration is invalid
    """
    # Parse defaults if provided
    defaults_dict = config_dict.get("defaults", {})
    defaults = _parse_step_config(defaults_dict, is_default=True) if defaults_dict else None

    # Parse datasets if provided
    datasets = {}
    for name, dataset_dict in config_dict.get("datasets", {}).items():
        datasets[name] = _parse_dataset_config(dataset_dict)

    # Get default dataset name
    default_dataset = config_dict.get("default_dataset")

    # Parse steps
    steps = []
    for i, step_dict in enumerate(config_dict.get("steps", [])):
        # Merge with defaults
        merged_dict = {**defaults_dict, **step_dict} if defaults else step_dict
        step = _parse_step_config(merged_dict, is_default=False)
        steps.append(step)

    config = PipelineConfig(
        steps=steps, defaults=defaults, datasets=datasets, default_dataset=default_dataset
    )

    # Validate configuration
    validator = PipelineConfigValidator(config)
    errors = validator.validate()

    if errors:
        error_msg = "Pipeline configuration validation failed:\n" + "\n".join(
            f"  - {err}" for err in errors
        )
        raise ValueError(error_msg)

    return config


def _parse_step_config(step_dict: dict, is_default: bool) -> PipelineStepConfig:
    """Parse a single pipeline step configuration."""
    # Parse transition criteria
    transition_dict = step_dict.get("transition_on", {})
    transition = TransitionCriteria(
        mode=transition_dict.get("mode", "epoch"),
        value=transition_dict.get("value"),
        metric=transition_dict.get("metric"),
        threshold=transition_dict.get("threshold"),
        max_epochs=transition_dict.get("max_epochs"),
    )

    # Parse optimization config (inline YAML only)
    optimization = None
    if "optimization" in step_dict:
        opt_dict = step_dict["optimization"]

        # Parse optimizers
        optimizers = {}
        for name, opt_cfg in opt_dict.get("optimizers", {}).items():
            optimizers[name] = OptimizerConfig(
                type=opt_cfg.get("type", "AdamW"),
                lr=opt_cfg.get("lr", 1e-5),
                betas=tuple(opt_cfg["betas"]) if "betas" in opt_cfg else None,
                momentum=opt_cfg.get("momentum"),
                weight_decay=opt_cfg.get("weight_decay", 0.0),
                eps=opt_cfg.get("eps", 1e-8),
                alpha=opt_cfg.get("alpha"),
                centered=opt_cfg.get("centered", False),
                amsgrad=opt_cfg.get("amsgrad", False),
                decoupled_weight_decay=opt_cfg.get("decoupled_weight_decay", True),
            )

        # Parse schedulers
        schedulers = {}
        for name, sched_cfg in opt_dict.get("schedulers", {}).items():
            schedulers[name] = SchedulerConfig(
                type=sched_cfg.get("type", "CosineAnnealingLR"),
                eta_min_factor=sched_cfg.get("eta_min_factor", 0.1),
                start_factor=sched_cfg.get("start_factor", 1.0),
                end_factor=sched_cfg.get("end_factor", 0.1),
                total_iters=sched_cfg.get("total_iters"),
                step_size=sched_cfg.get("step_size"),
                gamma=sched_cfg.get("gamma", 0.1),
                mode=sched_cfg.get("mode", "min"),
                factor=sched_cfg.get("factor", 0.1),
                patience=sched_cfg.get("patience", 10),
            )

        optimization = OptimizationConfig(optimizers=optimizers, schedulers=schedulers)

    # Create step config
    return PipelineStepConfig(
        name=step_dict.get("name", ""),
        n_epochs=step_dict.get("n_epochs", 0) if not is_default else 0,
        max_steps=step_dict.get("max_steps"),
        description=step_dict.get("description", ""),
        dataset=step_dict.get("dataset"),  # Dataset name
        train_vae=step_dict.get("train_vae", False),
        gan_training=step_dict.get("gan_training", False),
        use_lpips=step_dict.get("use_lpips", False),
        train_spade=step_dict.get("train_spade", False),
        train_diff=step_dict.get("train_diff", False),
        train_diff_full=step_dict.get("train_diff_full", False),
        use_ema=step_dict.get("use_ema", True),
        cfg_dropout_prob=step_dict.get("cfg_dropout_prob", 0.0),
        freeze=step_dict.get("freeze", []),
        unfreeze=step_dict.get("unfreeze", []),
        batch_size=step_dict.get("batch_size", 2),
        workers=step_dict.get("workers", 8),
        lr=step_dict.get("lr"),
        lr_min=step_dict.get("lr_min", 0.1),
        use_fp16=step_dict.get("use_fp16", False),
        initial_clipping_norm=step_dict.get("initial_clipping_norm", 1.0),
        kl_beta=step_dict.get("kl_beta", 0.0001),
        kl_warmup_steps=step_dict.get("kl_warmup_steps", 5000),
        kl_free_bits=step_dict.get("kl_free_bits", 0.0),
        lambda_adv=step_dict.get("lambda_adv", 0.5),
        lambda_lpips=step_dict.get("lambda_lpips", 0.1),
        mse_weight=step_dict.get("mse_weight", 0.1),
        r1_interval=step_dict.get("r1_interval", 16),
        r1_gamma=step_dict.get("r1_gamma", 5.0),
        instance_noise_std=step_dict.get("instance_noise_std", 0.01),
        instance_noise_decay=step_dict.get("instance_noise_decay", 0.9999),
        adaptive_weights=step_dict.get("adaptive_weights", True),
        optimization=optimization,
        transition_on=transition,
    )
