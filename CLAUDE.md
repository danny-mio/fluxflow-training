# CLAUDE.md - FluxFlow Training

## Project Overview

FluxFlow Training is a Python package for training text-to-image generation models, specifically VAE (Variational AutoEncoders) and Flow-based diffusion transformers. The package supports GAN losses, mixed precision training, checkpoint resumption, and streaming datasets.

**Version**: 0.1.0 | **License**: MIT | **Python**: 3.10+

## Quick Commands

```bash
# Setup
make install-dev              # Install with dev dependencies
pre-commit install            # Set up pre-commit hooks

# Testing
make test                     # Run all tests
pytest tests/unit/ -v         # Unit tests only
pytest tests/integration/ -v  # Integration tests only
pytest tests/unit/test_losses.py::test_d_hinge_loss -v  # Single test

# Code Quality
make lint                     # Run flake8, black --check, isort --check
make format                   # Auto-format with black + isort
pre-commit run --all-files    # Run all pre-commit hooks

# Training
fluxflow-train --config config.yaml --train_vae
make train CONFIG=config.yaml
```

## Architecture

```
src/fluxflow_training/
├── scripts/                  # CLI entry points
│   ├── train.py              # Main training script
│   ├── generate.py           # Image generation
│   └── generate_training_graphs.py
└── training/                 # Core training logic
    ├── vae_trainer.py        # VAE training with GAN losses
    ├── flow_trainer.py       # Flow model training (v-prediction)
    ├── checkpoint_manager.py # Save/load checkpoints
    ├── losses.py             # Loss functions (hinge, KL, MMD, R1)
    ├── utils.py              # EMA, FloatBuffer, device utils
    ├── optimizer_factory.py  # AdamW, Lion, SGD creation
    ├── scheduler_factory.py  # LR scheduler creation
    └── progress_logger.py    # Training metrics logging
```

## Key Dependencies

- **PyTorch 2.0+**: Core deep learning framework
- **accelerate**: Distributed training, mixed precision
- **transformers**: BERT text encoding
- **diffusers**: Diffusion model utilities
- **webdataset**: Streaming dataset support (TTI-2M)
- **fluxflow**: Core FluxFlow models (required dependency)

## Code Style

- **Python 3.10+** with type hints on public APIs
- **Black** formatting (line-length=100)
- **isort** imports (profile=black)
- **Docstrings**: Google-style for all public functions/classes
- **Naming**: snake_case (functions/vars), PascalCase (classes), UPPER_SNAKE (constants)
- **Max complexity**: 15 (flake8), functions < 50 lines
- **Imports order**: stdlib → third-party → local (blank line separated)

## Testing

Tests are in `tests/` with unit and integration subdirectories.

**Pytest Markers**:
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.gpu` - Requires GPU
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.requires_model` - Needs model checkpoint
- `@pytest.mark.requires_data` - Needs dataset

**Fixtures** (in `tests/conftest.py`): `device`, `temp_dir`, `mock_models`, `mock_dataloaders`, `random_image_tensor`, etc.

```bash
# Skip slow tests
pytest tests/ -v -m "not slow"

# With coverage
pytest tests/ -v --cov=fluxflow_training --cov-report=html
```

## Configuration

Training uses YAML configuration files (see `config.example.yaml`):

```yaml
model:
  vae_dim: 128
  feature_maps_dim: 128
  text_embedding_dim: 1024

data:
  data_path: "/path/to/images"
  captions_file: "/path/to/captions.txt"  # image_name<tab>caption
  img_size: 1024

training:
  n_epochs: 100
  batch_size: 1
  lr: 0.00001
  train_vae: true
  use_fp16: false

output:
  output_path: "outputs/flux"
  log_interval: 10
  checkpoint_save_interval: 50
```

CLI arguments override YAML settings.

## Important Files

- `pyproject.toml` - Package config, dependencies, entry points
- `Makefile` - Development targets
- `.flake8` - Linting config
- `pytest.ini` - Test config
- `.pre-commit-config.yaml` - Pre-commit hooks
- `docs/TRAINING_GUIDE.md` - Comprehensive training guide
- `CONTRIBUTING.md` - Contribution guidelines

## Common Patterns

### Loss Functions (`training/losses.py`)
```python
from fluxflow_training.training.losses import d_hinge_loss, g_hinge_loss, r1_penalty
```

### EMA for Model Stabilization (`training/utils.py`)
```python
from fluxflow_training.training.utils import EMA
ema = EMA(model, decay=0.999)
ema.update()
```

### Checkpoint Management (`training/checkpoint_manager.py`)
```python
from fluxflow_training.training.checkpoint_manager import CheckpointManager
manager = CheckpointManager(output_dir)
manager.save_checkpoint(pipeline, optimizer, scheduler, epoch, step)
```

## CI/CD

GitHub Actions runs on push/PR to main/develop:
1. Linting (flake8, black, isort)
2. Type checking (mypy)
3. Tests with coverage (pytest)
4. Coverage upload to codecov.io

Matrix: Python 3.10, 3.11, 3.12

## Development Workflow

1. Create feature branch
2. Make changes with type hints and docstrings
3. Run `make format` and `make lint`
4. Run `make test` (or specific tests)
5. Run `pre-commit run --all-files`
6. Commit with descriptive message
7. Push and create PR
