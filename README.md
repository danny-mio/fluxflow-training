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

**Package available on PyPI**: [fluxflow-training v0.1.1](https://pypi.org/project/fluxflow-training/)

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

- **VAE Training**: Train variational autoencoders with GAN losses and LPIPS perceptual loss
- **Flow Training**: Train flow-based diffusion transformers
- **Data Loading**: Efficient dataset handling with WebDataset support
- **Checkpointing**: Robust checkpoint management with resumability
- **Training Visualization**: Automatic diagram generation for loss curves and metrics
- **Optimizers**: Multiple optimizer support (AdamW, Lion, SGD)
- **Schedulers**: Various learning rate schedulers
- **Mixed Precision**: Accelerate training with automatic mixed precision

## Quick Start

### Training a Model

```bash
# Create a config file (see config.example.yaml)
fluxflow-train --config config.yaml

# With automatic diagram generation on each checkpoint
fluxflow-train --config config.yaml --generate_diagrams
```

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

## Package Contents

- `fluxflow_training.training` - Training logic and trainers
- `fluxflow_training.data` - Dataset implementations and transforms
- `fluxflow_training.scripts` - CLI scripts for training and generation

## Configuration

Training is configured via YAML files. See [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) for detailed configuration options.

Example:

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
  use_fp16: false

output:
  output_path: "outputs/flux"
  log_interval: 10
  checkpoint_save_interval: 50
  sample_sizes:  # Optional: generate samples at various sizes
    - 512
    - [768, 512]  # landscape
    - 1024
```

## Documentation

**Training Guide**: See [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) for:
- Detailed configuration options
- Dataset preparation
- Training strategies
- Troubleshooting

## Links

- [GitHub Repository](https://github.com/danny-mio/fluxflow-training)
- [Core Package](https://github.com/danny-mio/fluxflow-core)
- [Web UI](https://github.com/danny-mio/fluxflow-ui)
- [ComfyUI Plugin](https://github.com/danny-mio/fluxflow-comfyui)

## License

MIT License - see LICENSE file for details.
