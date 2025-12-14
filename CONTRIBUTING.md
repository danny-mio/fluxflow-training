# Contributing to FluxFlow

Thank you for your interest in contributing to FluxFlow! This guide will help you get started with development.

## Quick Start for Developers

### 1. Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/danny-mio/fluxflow-training.git
cd fluxflow-training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (automatic code quality checks)
pre-commit install
```

### 2. Development Workflow

#### Before Making Changes

```bash
# Create a new branch
git checkout -b feature/your-feature-name
```

#### During Development

```bash
# Format your code
make format

# Run linting
make lint

# Run type checking
mypy src/

# Run tests
make test
```

#### Before Committing

Pre-commit hooks will automatically run when you commit. To run them manually:

```bash
# Run all pre-commit checks
pre-commit run --all-files
```

### 3. Code Quality Standards

FluxFlow uses several tools to maintain code quality:

- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

All of these run automatically via pre-commit hooks, but you can also run them manually:

```bash
make format                    # Format code
make lint                      # Check linting
mypy src/                      # Check types
make test                      # Run tests
pre-commit run --all-files     # Run all checks
```

## Code Style Guidelines

### Python Code

- Follow PEP 8 (enforced by flake8)
- Use Black formatting (100 character line limit)
- Add type hints to public APIs
- Write docstrings for modules, classes, and functions
- Keep functions focused and small (< 50 lines ideally)

### Example

```python
"""Module docstring explaining the module's purpose."""

from typing import Optional

import logging

logger = logging.getLogger(__name__)


def process_data(input_path: str, output_path: Optional[str] = None) -> dict[str, int]:
    """
    Process data from input file and optionally save to output.

    Args:
        input_path: Path to input data file
        output_path: Optional path to save processed data

    Returns:
        Dictionary with processing statistics

    Raises:
        FileNotFoundError: If input_path doesn't exist
    """
    logger.info(f"Processing data from {input_path}")
    # Implementation here
    return {"processed": 42}
```

### Configuration Files

- Use YAML for configuration (not shell scripts for new configs)
- Add validation with Pydantic models
- Include examples in `config.example.yaml`

### Documentation

- Update README.md for user-facing changes
- Update TRAINING_GUIDE.md for new training parameters
- Keep documentation concise and practical
- Use code examples where helpful

## Testing

### Writing Tests

All new code should include tests:

```bash
# Create test file in appropriate directory
tests/unit/test_your_feature.py
tests/integration/test_your_workflow.py
```

### Test Structure

```python
"""Tests for your feature."""

import pytest
from fluxflow_training.your_module import YourClass


class TestYourClass:
    """Tests for YourClass."""

    def test_basic_functionality(self):
        """Test basic functionality works."""
        obj = YourClass()
        result = obj.method()
        assert result == expected

    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            obj = YourClass(invalid_param="bad")
```

### Running Tests

```bash
# All tests
make test

# Unit tests only
pytest tests/unit/ -v

# Skip slow tests
pytest tests/ -v -m "not slow"

# Specific test file
pytest tests/unit/test_your_feature.py -v

# Specific test function
pytest tests/unit/test_your_feature.py::test_function_name -v
```

### Testing Pipeline Mode

**Pipeline mode requires special testing** since it involves multi-step orchestration:

#### Quick Pipeline Test (Recommended)

Use `max_steps` to run a few batches per step:

```yaml
# test_pipeline.yaml
data:
  data_path: "tests/fixtures/images"
  captions_file: "tests/fixtures/captions.txt"

training:
  batch_size: 1
  output_path: "tests/output/pipeline_test"
  
  pipeline:
    steps:
      - name: "step1"
        n_epochs: 1
        max_steps: 5  # Only run 5 batches
        train_vae: true
        
      - name: "step2"
        n_epochs: 1
        max_steps: 5
        train_vae: true
        gan_training: true
```

**Run:**
```bash
fluxflow-train --config tests/test_pipeline.yaml
```

**Expected:** Training completes in ~30 seconds, creates step-specific checkpoints.

#### Testing Checklist for Pipeline Changes

When modifying pipeline orchestration code:

- [ ] Test with minimal config (1 epoch, max_steps=5)
- [ ] Test step transitions (verify step 2 loads step 1 checkpoint)
- [ ] Test resume (interrupt training, restart, verify continues from correct step)
- [ ] Test freeze flags (`freeze_vae`, `freeze_flow`, `freeze_text_encoder`)
- [ ] Test loss-based stop conditions (if applicable)
- [ ] Test step-specific metrics files created
- [ ] Test diagram generation for each step
- [ ] Verify all unit tests pass: `pytest tests/unit/ -v`

#### Example Test Script

```bash
#!/bin/bash
# test_pipeline_quick.sh - Quick pipeline smoke test

set -e

# Cleanup
rm -rf tests/output/pipeline_test

# Run pipeline
fluxflow-train --config tests/fixtures/pipeline_minimal.yaml

# Verify outputs
test -f tests/output/pipeline_test/flxflow_step_step1_final.safetensors || exit 1
test -f tests/output/pipeline_test/flxflow_step_step2_final.safetensors || exit 1
test -f tests/output/pipeline_test/graph/training_metrics_step1.jsonl || exit 1
test -f tests/output/pipeline_test/graph/training_metrics_step2.jsonl || exit 1

echo "✅ Pipeline smoke test passed"
```

#### Testing GAN-Only Mode

When testing `train_reconstruction` parameter:

```yaml
# test_gan_only.yaml
training:
  pipeline:
    steps:
      - name: "gan_only"
        n_epochs: 1
        max_steps: 10
        train_vae: true
        gan_training: true
        train_spade: true
        train_reconstruction: false  # GAN-only mode
```

**Verify:**
- No reconstruction loss logged (`loss_recon` should be absent)
- GAN losses present (`loss_gen`, `loss_disc`)
- Encoder gradients flow (check with gradient inspection if needed)

#### Integration Test Example

For new pipeline features, add integration tests:

```python
# tests/integration/test_pipeline_orchestration.py
"""Integration tests for pipeline orchestration."""

import pytest
import yaml
from pathlib import Path


def test_two_step_pipeline(tmp_path):
    """Test basic two-step pipeline completes."""
    config = {
        "data": {
            "data_path": "tests/fixtures/images",
            "captions_file": "tests/fixtures/captions.txt",
        },
        "training": {
            "batch_size": 1,
            "output_path": str(tmp_path),
            "pipeline": {
                "steps": [
                    {"name": "step1", "n_epochs": 1, "max_steps": 3, "train_vae": True},
                    {"name": "step2", "n_epochs": 1, "max_steps": 3, "train_vae": True, "gan_training": True},
                ]
            }
        }
    }
    
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    # Run training
    from fluxflow_training.scripts.train import main
    main(["--config", str(config_path)])
    
    # Verify outputs
    assert (tmp_path / "flxflow_step_step1_final.safetensors").exists()
    assert (tmp_path / "flxflow_step_step2_final.safetensors").exists()
    assert (tmp_path / "graph" / "training_metrics_step1.jsonl").exists()
    assert (tmp_path / "graph" / "training_metrics_step2.jsonl").exists()
```

## Project Structure

```
fluxflow-training/
├── src/fluxflow_training/     # Main package source
│   ├── scripts/               # CLI entry points (train, generate)
│   └── training/              # Core training logic
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── docs/                      # Documentation
├── pyproject.toml             # Package configuration
├── Makefile                   # Development targets
└── config.example.yaml        # Configuration template
```

Install with pip:
- `pip install -e ".[dev]"` - Development installation with dev dependencies
- Note: Not yet published to PyPI

## Pull Request Process

1. **Create a branch** from `develop`
2. **Make your changes** with tests and documentation
3. **Run all checks**: `pre-commit run --all-files && make test`
4. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: brief description

   Detailed explanation of what changed and why.

   - Specific change 1
   - Specific change 2"
   ```
5. **Push and create PR** on GitHub
6. **Address review feedback** if any

## Common Tasks

### Adding a New Feature

1. Create feature branch: `git checkout -b feature/name`
2. Add implementation in appropriate module
3. Add type hints and docstrings
4. Add tests for the feature
5. Update documentation if user-facing
6. Run `pre-commit run --all-files && make test` to verify quality
7. Commit and create PR

### Fixing a Bug

1. Create bugfix branch: `git checkout -b fix/issue-description`
2. Add a test that reproduces the bug
3. Fix the bug
4. Verify the test passes
5. Run `pre-commit run --all-files && make test`
6. Commit and create PR

### Adding Dependencies

1. Add to appropriate section in `pyproject.toml`:
   - Core dependencies → `dependencies`
   - Dev dependencies → `optional-dependencies.dev`
2. Update requirements.txt if needed for backward compatibility
3. Document why the dependency is needed in your PR

## Requesting Contributor Access

### Public Contributors (No Access Needed)

Anyone can contribute via pull requests! Just:

1. Fork the repository
2. Make changes in your fork
3. Submit a pull request to the `develop` branch

No special permissions needed!

### Requesting Direct Repository Access

If you want to contribute regularly:

1. **Start by contributing** - Submit 1-3 quality PRs first
2. **Open a discussion** at https://github.com/danny-mio/fluxflow-training/discussions
3. **Use title**: `Request: Contributor Access for [Your Name]`
4. **Include**: GitHub username, merged PRs, contribution areas, availability, experience

Repository owner reviews requests within 7 days.

## Security Issues

For security vulnerabilities, see [SECURITY.md](SECURITY.md).

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with reproduction steps
- **Features**: Open a GitHub Issue to discuss before implementing

## Code of Conduct

- Be respectful and professional
- Focus on constructive feedback
- Welcome newcomers and help them learn
- Keep discussions focused on technical merit

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to FluxFlow!** 🎉

## Testing Memory-Intensive Features

When contributing optimizations or features that affect memory usage:

### Memory Testing Checklist

- [ ] Test on target GPU (e.g., A6000 48GB)
- [ ] Monitor VRAM with `nvidia-smi -l 1` during training
- [ ] Record peak VRAM usage for documentation
- [ ] Test with full config (GAN+LPIPS+SPADE) if applicable
- [ ] Verify no OOM on 48GB GPU with batch_size=4
- [ ] Document memory savings if optimization
- [ ] Update TRAINING_GUIDE.md "Memory Requirements" section

### Example Memory Test Script

```bash
# test_memory.sh
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > memory_log.txt &
NVIDIA_PID=$!

fluxflow-train --config test_config.yaml

kill $NVIDIA_PID
echo "Peak VRAM: $(sort -n memory_log.txt | tail -1) MB"
```

### Reporting Memory Issues

When reporting OOM or memory issues, include:
1. GPU model and VRAM size
2. Full config file
3. Exact command run
4. Peak VRAM usage (from `nvidia-smi`)
5. FluxFlow version (`pip show fluxflow-training`)
6. CUDA version (`nvcc --version` or `nvidia-smi`)

Example:
```
GPU: NVIDIA A6000 (48GB)
Peak VRAM: 47.4GB (triggered OOM)
Config: batch_size=4, vae_dim=128, gan_training=true, use_lpips=true, train_spade=true
Version: fluxflow-training 0.2.1
CUDA: 12.1
```

