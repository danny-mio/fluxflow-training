# FluxFlow Training

Training tools and scripts for FluxFlow text-to-image generation models.

## Installation

### Production Install

```bash
pip install fluxflow-training
```

**What gets installed:**
- `fluxflow-training` - Training scripts and configuration tools
- `fluxflow` core package (automatically installed as dependency)
- CLI commands: `fluxflow-train`, `fluxflow-generate`

### Development Install

```bash
git clone https://github.com/danny-mio/fluxflow-training.git
cd fluxflow-training
pip install -e ".[dev]"
```

---

## 🚧 Training Status

**Models Currently In Training**: FluxFlow is actively training models following the systematic [TRAINING_VALIDATION_PLAN.md](https://github.com/danny-mio/fluxflow-core/blob/main/TRAINING_VALIDATION_PLAN.md).

**Current Phase**: Phase 1 - VAE Training (Weeks 1-4)

**Progress**:
- 🔄 Bezier VAE training in progress
- ⏳ ReLU baseline VAE pending
- ⏳ Flow models pending VAE completion

**When Available**: Trained checkpoints and empirical performance metrics will be published to [MODEL_ZOO.md](https://github.com/danny-mio/fluxflow-core/blob/main/MODEL_ZOO.md) upon validation completion.

**Note**: All performance claims in documentation are theoretical targets pending empirical validation.

---

## Hardware Requirements for Training

**Current Setup** (December 2025 validation):
- **GPU**: NVIDIA A6000 (48GB VRAM) via Paperspace
- **Constraint**: Service interrupts every 6 hours
- **Solution**: Automatic checkpoint/resume (every 30-60 min)

**Alternative Options**:
- Local: 1× RTX 4090 (24GB) or 2× RTX 3090 (24GB each)
- Cloud: AWS p3.2xlarge (V100 16GB), GCP A100 (40GB)

**Minimum Requirements**:
- 16GB VRAM for VAE training (batch size 2-4)
- 24GB VRAM for Flow training (batch size 1-2)
- 48GB VRAM enables larger batches (batch size 8+)

**Cost Comparison**:
| Platform | GPU | $/hr | Est. Total (500 hrs) |
|----------|-----|------|---------------------|
| **Paperspace** | **A6000 48GB** | **$0.76** | **~$456** ✓ |
| AWS | p3.2xlarge (V100 16GB) | $3.06 | ~$1,530 |
| GCP | A100 40GB | $3.67 | ~$1,835 |
| Lambda Labs | A100 40GB | $1.10 | ~$550 |

---

### Pre-download LPIPS Weights (Optional)

The training uses LPIPS for perceptual loss, which requires VGG16 weights (~528MB). To pre-download:

```bash
python -c "import lpips; lpips.LPIPS(net='vgg')"
```

Weights will be cached in `~/.cache/torch/hub/checkpoints/`. If not pre-downloaded, they'll download automatically on first training run.

## Features

### Core Training Capabilities

- **🎯 Pipeline Training Mode** (NEW in v0.2.0)
  - Multi-step sequential training with independent configs per step
  - Per-step freeze/unfreeze of model components
  - Loss-threshold transitions with early stopping
  - Full checkpoint resume from any step/epoch/batch
  - See [PIPELINE_ARCHITECTURE.md](docs/PIPELINE_ARCHITECTURE.md)

- **🎨 GAN-Only Training Mode** (NEW in v0.2.0)
  - Train encoder/decoder with adversarial loss only (no reconstruction)
  - Spatial conditioning (SPADE) without pixel-perfect reconstruction
  - Faster training with focused gradient flow

- **VAE Training**
  - Variational autoencoders with GAN losses
  - LPIPS perceptual loss support
  - SPADE spatial conditioning
  - KL divergence with beta warmup and free bits

- **Flow Training**
  - Flow-based diffusion transformers
  - Text-to-image generation
  - Conditional and unconditional modes

### Training Infrastructure

- **Data Loading**: Efficient dataset handling with WebDataset support for large-scale training
- **Checkpointing**: Robust checkpoint management with automatic resume from interruptions
- **Training Visualization**: Automatic diagram generation for loss curves and metrics
- **Optimizers**: Multiple optimizer support (AdamW, Adam, SGD, RMSprop)
- **Schedulers**: Various learning rate schedulers (CosineAnnealing, StepLR, ExponentialLR, ReduceLROnPlateau)
- **Mixed Precision**: Accelerate training with automatic mixed precision (fp16)

### Monitoring & Logging

- **Enhanced Console Output** (v0.2.0)
  - Real-time batch timing (seconds/batch)
  - Comprehensive loss display (VAE, KL, G, D, LPIPS)
  - Step/epoch/batch progress tracking
  
- **Step-Specific Metrics** (Pipeline Mode)
  - Separate JSONL files per training step
  - Automatic graph generation per step
  - Full training history preservation

- **Mid-Epoch Sample Generation** (v0.2.0)
  - Samples generated at configurable intervals
  - Structured naming with step/epoch/batch numbers
  - Prevents overwrites with unique identifiers

## Quick Start

### Basic Training

```bash
# Create a config file (see config.example.yaml)
fluxflow-train --config config.yaml

# With automatic diagram generation
fluxflow-train --config config.yaml --generate_diagrams
```

### Pipeline Training (NEW)

Multi-step training for hypothesis testing and staged optimization:

```yaml
training:
  pipeline:
    steps:
      # Step 1: VAE warmup
      - name: "vae_warmup"
        n_epochs: 10
        train_vae: true
        gan_training: false
        freeze:
          - flow_processor
          - text_encoder
      
      # Step 2: Add GAN
      - name: "vae_gan"
        n_epochs: 20
        train_vae: true
        gan_training: true
        use_lpips: true
        
      # Step 3: Flow training
      - name: "flow"
        n_epochs: 30
        train_vae: false      # Freeze VAE
        train_diff: true      # Train flow
```

**Run**:
```bash
fluxflow-train --config pipeline_config.yaml
```

**Validate config before training**:
```bash
fluxflow-train --config pipeline_config.yaml --validate-pipeline
```

See [PIPELINE_ARCHITECTURE.md](docs/PIPELINE_ARCHITECTURE.md) for complete documentation.

### GAN-Only Training (NEW)

Train with adversarial loss only (no pixel-level reconstruction):

```yaml
training:
  pipeline:
    steps:
      - name: "gan_only"
        n_epochs: 20
        train_vae: false          # Skip reconstruction loss
        gan_training: true        # Train GAN discriminator
        train_spade: true         # With spatial conditioning
        use_lpips: false
```

**Use cases:**
- Spatial structure learning without pixel constraints
- Faster training (no reconstruction computation)
- SPADE conditioning experiments

### Generating Images

```bash
# Create a directory with .txt files containing prompts
mkdir prompts
echo "a beautiful sunset over mountains" > prompts/sunset.txt

# Generate images
fluxflow-generate \
    --model_checkpoint path/to/checkpoint.safetensors \
    --text_prompts_path prompts/ \
    --output_path outputs/
```

### Visualizing Training Progress

```bash
# Training metrics are automatically logged to outputs/graph/training_metrics.jsonl
# In pipeline mode: outputs/graph/training_metrics_{stepname}.jsonl

# Generate diagrams from logged metrics
python src/fluxflow_training/scripts/generate_training_graphs.py outputs/

# Diagrams are saved to outputs/graph/:
# - training_losses.png (VAE, Flow, Discriminator, Generator, LPIPS)
# - kl_loss.png (KL divergence with beta warmup)
# - learning_rates.png (LR schedules)
# - batch_times.png (training speed)
# - training_overview.png (combined overview)
# - training_summary.txt (statistics)
```

## Console Output Examples

### Pipeline Training Mode
```
PIPELINE STEP 1/3: vae_warmup
Description: Warmup VAE without GAN
Duration: 10 epochs
================================================================================

[00:15:23] Step vae_warmup (1/3) | Epoch 5/10 | Batch 127/500 | VAE: 0.0234 | KL: 12.45 | 3.2s/batch
```

### GAN Training Mode
```
[00:15:23] Step vae_gan (2/3) | Epoch 5/20 | Batch 127/500 | VAE: 0.0234 | KL: 12.45 | G: 0.156 | D: 0.089 | LPIPS: 0.0812 | 3.2s/batch
```

**Metrics Shown**:
- `VAE` - Reconstruction loss (L1 + MSE)
- `KL` - KL divergence
- `G` - GAN generator loss (when `gan_training: true`)
- `D` - GAN discriminator loss (when `gan_training: true`)
- `LPIPS` - Perceptual loss (when `use_lpips: true`)
- `Xs/batch` - Average batch processing time

## Sample Naming Convention

### Pipeline Mode (v0.2.0)

**Mid-epoch samples**:
```
vae_warmup_001_005_00127_abc123-original.webp
{stepname}_{step}_{epoch}_{batch}_{hash}-{suffix}.webp
```

**End-of-epoch samples**:
```
vae_warmup_001_005_abc123-original.webp
{stepname}_{step}_{epoch}_{hash}-{suffix}.webp
```

**Suffixes**:
- `-original` or `_ns_i` - Input image
- `-nr_o` - VAE reconstruction (no random sampling)
- `_ns_o` - VAE reconstruction (with sampling)
- `-ctx` - Context vector visualization
- `-nc` - Non-context visualization

## Package Contents

- `fluxflow_training.training` - Training logic and trainers
  - `pipeline_orchestrator.py` - Multi-step pipeline execution
  - `pipeline_config.py` - Pipeline configuration and validation
  - `vae_trainer.py` - VAE/GAN training logic
  - `flow_trainer.py` - Flow model training
  - `checkpoint_manager.py` - Checkpoint save/resume
  
- `fluxflow_training.data` - Dataset implementations and transforms
  - `datasets.py` - Image/caption datasets with WebDataset support
  - `transforms.py` - Data augmentation pipelines
  
- `fluxflow_training.scripts` - CLI scripts for training and generation
  - `train.py` - Main training script (legacy + pipeline mode)
  - `generate_training_graphs.py` - Visualization generation

## Configuration

Training is configured via YAML files. See [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) for detailed configuration options.

### Basic Configuration Example

```yaml
model:
  vae_dim: 128
  feature_maps_dim: 128
  text_embedding_dim: 1024

data:
  data_path: "/path/to/images"
  captions_file: "/path/to/captions.txt"  # image_name<tab>caption
  fixed_prompt_prefix: null  # Optional: e.g., "style anime" to prepend to all prompts
  img_size: 1024
  reduced_min_sizes: null  # Optional: [128, 256, 512]

training:
  n_epochs: 100
  batch_size: 1
  lr: 0.00001
  train_vae: true
  gan_training: true
  use_lpips: true
  use_fp16: false

output:
  output_path: "outputs/flux"
  log_interval: 10
  checkpoint_save_interval: 50
  samples_per_checkpoint: 1  # Generate samples at each checkpoint
  sample_sizes:  # Optional: generate samples at various sizes
    - 512
    - [768, 512]  # landscape
    - 1024
```

### Pipeline Configuration Example

See `test_pipeline_minimal.yaml` or `config.example.yaml` for complete examples.

**Key difference**: Pipeline configs use `training.pipeline.steps` instead of flat `training` config.

```yaml
training:
  batch_size: 4
  workers: 8
  
  pipeline:
    steps:
      - name: "step1"
        n_epochs: 10
        train_vae: true
        # ... step-specific config
```

## Documentation

### Comprehensive Guides

- **[PIPELINE_ARCHITECTURE.md](docs/PIPELINE_ARCHITECTURE.md)** - Pipeline training mode documentation
  - Configuration reference
  - Checkpoint format
  - Sample naming conventions
  - Troubleshooting

- **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** - Complete training guide
  - Detailed configuration options
  - Dataset preparation
  - Training strategies
  - Best practices

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guide
  - Setting up development environment
  - Running tests
  - Code style guidelines
  - Pipeline config testing

### Example Configurations

- `config.example.yaml` - Complete configuration template with all options
- `test_pipeline_minimal.yaml` - Minimal working pipeline example
- `config.example.sh` - Environment setup script

## What's New in v0.2.0

### Major Features

- **Pipeline Training Mode** - Multi-step sequential training with per-step configs
- **GAN-Only Mode** - Train with adversarial loss only (no reconstruction)
- **Enhanced Logging** - Batch timing, step-specific metrics, correct GAN loss display
- **Mid-Epoch Samples** - Configurable sample generation with batch numbers

### Bug Fixes

- Fixed encoder gradient flow in GAN-only mode
- Fixed GAN loss logging (generator/discriminator keys)
- Fixed EMA creation for GAN-only mode
- Fixed metrics logging to respect training modes
- Fixed sample generation to prevent overwrites

### Breaking Changes

**Config format**: Pipeline configs must use `steps:` wrapper:

```yaml
# OLD (will error)
training:
  pipeline:
    - name: "step1"

# NEW (required)
training:
  pipeline:
    steps:
      - name: "step1"
```

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

## Links

- [GitHub Repository](https://github.com/danny-mio/fluxflow-training)
- [Core Package](https://github.com/danny-mio/fluxflow-core)
- [Web UI](https://github.com/danny-mio/fluxflow-ui)
- [ComfyUI Plugin](https://github.com/danny-mio/fluxflow-comfyui)
- [PyPI Package](https://pypi.org/project/fluxflow-training/) (coming soon - not yet published)

## License

MIT License - see LICENSE file for details.
