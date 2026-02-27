# Changelog

All notable changes to FluxFlow Training will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

_No unreleased changes._

## [0.8.1] - 2026-02-27

### Fixed
- **v-prediction loss target**: Flow trainer was computing loss against clean x0 despite `prediction_type="v_prediction"`. Now correctly computes `v = alpha_t * noise - sigma_t * x0` using `alphas_cumprod[t]`.
- **Context dim train/inference mismatch**: Previously only VAE dims were noised during training while context dims were passed clean, but inference denoises all 133 dims from noise. Training now noises all dims uniformly to match inference. The trivial x0-reconstruction ctx_loss is replaced with v-prediction over all dims.
- **Context dim loss scale**: VAE dims (~unit Gaussian) and context dims (~0.1–0.5 range) are normalised independently to prevent a ~10x loss imbalance. A `ctx_loss_weight` knob (default `1.0`) allows tuning.
- **Gradient clipping**: Replaced a broken adaptive clipping formula (`min(clip_norm, norm*1.5)`) with a straightforward `clip_grad_norm_` call. Removes the redundant manual norm loop.

### Added
- **`ctx_loss` console logging**: Context dim loss is now shown in the training progress line (` | Ctx: 0.0123`) alongside the flow loss.
- **`ctx_loss` graph curve**: Context dim loss is plotted as a separate curve in `training_losses.png` and the combined overview.
- **`ctx_loss_weight` config param**: Allows tuning the relative weight of the context-dim v-prediction loss in the flow training config.

## [0.8.0] - 2026-02-21

### Changed
- **Updated `fluxflow` dependency** to `>=0.8.0`
  - Enables training and inference with v0.8.0 pillar-attention flow architecture
  - `FlowTrainer` and pipeline are unchanged; version routing is handled by `load_versioned_checkpoint()`
- Version bumped to 0.8.0
- v0.8.0 pillar-attention flow architecture now supported via `model_version: "0.8.0"` in training config

## [0.7.1] - 2026-02-17

### Features
- **VAE loss component toggles**: Add configurable enable/disable for KL divergence, color statistics, histogram matching, contrast regularization, and coarseness losses
- **Coarseness loss**: New per-channel texture loss that matches local patch variance distributions between predicted and target images

### Fixes
- **Normalize timesteps**: Base normalization on the active window (`start_step` to `num_train_timesteps`)
- **Scheduler first step**: Correct `_first_step` handling so the scheduler advances properly
- **Context optimizer handling**: Prepare/step predictor optimizer safely with accelerator and reinit on dim changes

## [0.7.0] - 2026-02-12

### Fixed
- **Restore R1 penalty gradients**: Enable gradients on noisy real images before discriminator R1 regularization
- **Fix gradient accumulation**: Only zero grads at the start of accumulation windows in FlowTrainer
- **Tokenizer compatibility**: Add `batch_encode_plus` fallback for newer Transformers

### Changed
- **Stabilize dataset tests**: Ensure mock tokenizer returns tensor encodings via `__call__`

## [0.5.1] - 2025-12-24

### Fixed
- **Fixed pipeline training epoch switching**: Added missing break condition in batch processing loop to prevent infinite epochs
- **Fixed resume logic for invalid batch positions**: Added automatic advancement to next step/epoch when resume batch index exceeds epoch boundaries
- **Prevented training continuation beyond dataset bounds**: Pipeline training now properly stops at expected epoch boundaries

## [0.5.0] - 2025-12-23

### Changed
- **Updated fluxflow dependency** to `>=0.5.0,<0.6.0`
  - Aligns with fluxflow-core v0.5.0 release
  - Includes gradient checkpointing compatibility fixes
  - Bezier activation optimizations (JIT disabled for checkpoint compat)
  - Baseline model architecture support
  - Enhanced documentation and system requirements

## [0.4.0] - 2025-12-17

### Added

#### CFG-Enabled Training Sample Generation
- **Training samples now use CFG by default** when generating flow model samples
  - Automatically enables `use_cfg=True` with `guidance_scale=5.0`
  - Provides better preview quality during training
  - Matches inference-time generation quality
  - Only applies to flow training (`train_diff` or `train_diff_full`)
  - **Requires**: fluxflow-core with CFG sample generation support

#### Multi-Dataset Pipeline Support
- **Define multiple named datasets** for different pipeline steps
  - Support for both local and webdataset sources in same pipeline
  - Per-dataset configuration: `batch_size`, `workers`, image folders, URLs
  - Assign specific datasets to individual steps via `dataset` field
  - Optional `default_dataset` for steps without explicit assignment
- **Use cases**:
  - Progressive training: High-res local → Low-res webdataset
  - Domain-specific: Train VAE on portraits, Flow on landscapes
  - Resource optimization: Local SSD for warmup, cloud storage for main training
- **Files**: `src/fluxflow_training/training/pipeline_config.py` (DatasetConfig, parsing, validation)
- **Documentation**: `docs/MULTI_DATASET_TRAINING.md` (285 lines with examples)
- **Example**: `examples/multi_dataset_pipeline.yaml`

#### Auto-Create Missing Models in Pipeline Mode
- **Automatic model initialization** when transitioning between pipeline steps
  - Prevents crashes when moving from VAE → Flow training
  - Auto-creates: `flow_processor`, `text_encoder`, `compressor`, `expander`, `D_img` (discriminator)
  - Uses default parameters from args (`vae_dim`, `feature_maps_dim`, `text_embedding_dim`)
  - Logs warnings when models are auto-created
  - Moves models to correct device automatically
- **User impact**: Pipeline mode now more resilient; no manual model initialization required

#### Model Validation Before Training
- **Pre-flight validation** checks required models exist before creating trainers
- **Clear error messages** listing missing models if validation fails
- Prevents cryptic AttributeError crashes during training
- **Files**: `src/fluxflow_training/training/pipeline_orchestrator.py`

### Changed
- **Added 21 comprehensive unit tests** for multi-dataset pipeline
  - DatasetConfig dataclass tests (3 tests)
  - Dataset parsing tests for local + webdataset (4 tests)
  - Step dataset assignment tests (3 tests)
  - Dataset validation tests (9 tests)
  - Backward compatibility tests (2 tests)
- **File**: `tests/unit/test_pipeline_multi_dataset.py`
- **Major TRAINING_GUIDE.md improvements** for YAML-first configuration
  - Added "Configuration Methods" section comparing YAML vs CLI approaches
  - Rewrote Quick Start with dual paths: "CLI Quick Test" vs "YAML Config (Production)"
  - Clear recommendation: YAML config for production, CLI for quick tests only
  - Feature comparison table showing YAML advantages
  - Eliminates confusion about external JSON optimizer configs
  - Emphasizes inline YAML optimizer configuration in pipeline mode
  - **Impact**: Users now understand YAML is the recommended production approach
- Added comprehensive multi-dataset training guide with use cases, examples, troubleshooting
- Added example pipeline configuration with multiple datasets

### Fixed

#### Logging and Sampling Bugs
- **CRITICAL: Missing JSONL records on crash/interrupt**
  - Added `f.flush()` to `progress_logger.log_metrics()` to force immediate disk writes
  - Prevents data loss when training is interrupted or crashes
  - **Impact**: All metrics are now guaranteed to be written to disk immediately
  - **Files**: `src/fluxflow_training/training/progress_logger.py:189`

- **VAE snapshots generated during Flow-only training**
  - Fixed `safe_vae_sample()` being called during pure Flow training (no VAE, no GAN, no SPADE)
  - Now generates VAE samples when encoder/decoder is being trained: `train_vae=True` OR `gan_training=True` OR `train_spade=True`
  - Correctly handles all encoder/decoder training modes:
    - VAE mode: Reconstruction loss training
    - GAN-only mode: Adversarial loss training without reconstruction
    - SPADE mode: Decoder SPADE conditioning training
  - **Impact**:
    - Eliminates confusing VAE samples during Flow-only training
    - Preserves samples for all encoder/decoder training modes
    - Reduces I/O overhead (~2-5 seconds per checkpoint for multi-image test sets)
    - Sample generation now accurately reflects active training modes

- **Sample generation decoupled from checkpointing**
  - Sample generation now triggered by `sample_interval` based on `global_step` (independent of checkpoint frequency)
  - Ensures consistent sample frequency across entire training run
  - Prevents missed samples when checkpoint interval doesn't align with sample needs
  - **Impact**: More reliable monitoring of training progress via samples

- **Linting errors** in pipeline configuration (trailing whitespace)
- **Pre-commit hooks** now enforced (flake8, black, pytest)

## [0.3.1] - 2025-12-13

### Changed
- **Updated fluxflow dependency** from `>=0.3.0` to `>=0.3.1`
  - Aligns with fluxflow-core v0.3.1 release
  - Note: v0.3.0 skipped due to release coordination issues

## [0.3.0] - 2025-12-12

### Added

#### Classifier-Free Guidance (CFG) Support
- **Training-time CFG implementation** with dropout-based conditioning
  - New `cfg_dropout_prob` parameter (default: 0.0) for CFG training
  - Randomly drops text conditioning during training to enable CFG inference
  - Typical values: 0.10-0.15 for balanced guidance control
- **CFG inference utilities** in `cfg_inference.py`
  - `generate_with_cfg()` function for dual-pass sampling
  - `guidance_scale` parameter (1.0-15.0) to control conditioning strength
  - Negative prompts for better control over unwanted features
- **CFG helper functions** in `cfg_utils.py`
  - `should_drop_text_conditioning()` - dropout logic
  - `create_cfg_latents()` - batch preparation for dual-pass
  - `apply_cfg_guidance()` - noise prediction combination
- **Comprehensive test suite**: 212 tests covering training, inference, and utilities
- **Memory validated**: CFG adds negligible overhead (<1 MB)

### Fixed

#### Memory Optimizations
- **CRITICAL FIX #1**: Removed LPIPS gradient checkpointing that caused OOM at 47.4GB on 48GB GPUs
  - Issue: LPIPS perceptual loss used gradient checkpointing, causing memory spikes
  - Impact: Training would OOM even on A6000 48GB with full config (GAN+LPIPS+SPADE)
  - Fix: Disabled gradient checkpointing in LPIPS (commit: 05196e7)
  - Result: Reduced LPIPS memory overhead by ~3-5GB

- **CRITICAL FIX #2**: Removed dataloader prefetch_factor causing memory overhead
  - Issue: DataLoader prefetch_factor=2 pre-loaded batches into VRAM
  - Impact: Added ~4-8GB memory overhead, contributed to OOM
  - Fix: Set prefetch_factor=None (commit: 14a24b8)
  - Result: Immediate memory reduction, training more stable

- **CRITICAL FIX #3**: Added aggressive CUDA cache clearing (commit: 8582cfb)
  - Clear cache before VAE backward pass
  - Clear cache after checkpoint save
  - Clear cache every 10 batches
  - Result: Prevents memory fragmentation, frees "reserved but unallocated" memory

#### Gradient & Training Fixes
- **R1 Penalty Gradient Fix**: Fixed R1 penalty gradient computation
  - Issue: R1 penalty wasn't computing gradients correctly, causing memory leaks
  - Impact: Discriminator training unstable, memory usage grew over time
  - Fix: Proper `torch.autograd.grad()` usage with `create_graph=True`
  - Result: Stable discriminator training, no memory leaks

### Changed

**VRAM Usage by Configuration** (A6000 48GB):
- VAE only (no GAN): ~18-22GB VRAM
- VAE + GAN: ~25-30GB VRAM
- VAE + GAN + LPIPS: ~28-35GB VRAM (after fixes)
- VAE + GAN + LPIPS + SPADE: ~35-42GB VRAM (after fixes)
- Peak observed (before fixes): 47.4GB → OOM
- Peak observed (after fixes): ~42GB → stable

## [0.2.1] - 2024-12-09

### Added

#### Pipeline Training Mode (NEW)
- **Multi-step sequential training** with per-step configuration
  - Define training stages in YAML config (warmup → GAN → flow, etc.)
  - Each step has its own: epochs, training modes, optimizers, schedulers
  - Automatic step detection and orchestration
- **Per-step freeze/unfreeze** for selective component training
  - `freeze_vae`, `freeze_flow`, `freeze_text_encoder` per step
  - Gradients automatically disabled for frozen models
- **Loss-threshold transitions** for adaptive training
  - Exit step when loss reaches target (e.g., `loss_recon < 0.01`)
  - Automatic progression to next step
- **Inline optimizer/scheduler configs** per step
  - Different optimizers per step (e.g., Adam warmup → Lion training)
  - Full per-model hyperparameter control via JSON config files
- **Per-step checkpoints and metrics**
  - Step-specific checkpoints: `flxflow_step_<name>_final.safetensors`
  - Step-specific metrics: `training_metrics_<step_name>.jsonl`
  - Step-specific diagrams: `training_losses_<step_name>.png`
- **Full resume support** mid-pipeline
  - Automatically loads last completed step
  - Preserves optimizer/scheduler/EMA states across steps

#### GAN-Only Training Mode (NEW)
- **`train_reconstruction` parameter** (default: `true`)
  - Set to `false` to train encoder/decoder with adversarial loss only
  - No pixel-level reconstruction loss computed
  - Use case: SPADE conditioning without reconstruction overhead
- **Integrated with pipeline mode**
  - Example: GAN-only warmup → full VAE+GAN training

#### WebDataset Optimizations
- **Reduced shuffle/shard buffering** for faster startup
  - `shardshuffle=10` (was 100) - reduced shard buffer
  - `.shuffle(100)` (was 1000) - reduced sample buffer
  - `workers: 1` recommended for streaming datasets
  - Result: First batch appears in seconds instead of minutes
- **WebDataset format parameters** in config and CLI
  - `webdataset_image_key` (e.g. "jpg", "png")
  - `webdataset_label_key` (e.g. "json")
  - `webdataset_caption_key` (e.g. "prompt", "caption")
  - Enables support for any HuggingFace WebDataset format

#### Stability Improvements
- **EMA (Exponential Moving Average)** for flow training to stabilize training and improve generation quality
  - Tracks `flow_processor` and `text_encoder` parameters
  - Configurable via `ema_decay` parameter (default: 0.9999)
- **NaN/Inf safety checks** in both VAE and Flow trainers
  - Automatic gradient zeroing on NaN detection
  - Prevents training crashes from numerical instability
  - Detailed logging for debugging

#### Loss Functions
- **LPIPS perceptual loss** (VGG-based, frozen network)
  - Significantly improves perceptual quality (expected LPIPS: 0.15 → 0.08)
  - Configurable via `use_lpips` and `lambda_lpips` parameters
  - Dependency: `lpips>=0.1.4`
- **Frequency-aware reconstruction loss**
  - Explicitly preserves high-frequency details and textures
  - Separate low/high frequency loss weighting
- **Text-image alignment loss** for flow training *(disabled by default, see Removed section)*
  - Cosine similarity between image and text features
  - Requires matching embedding dimensions (currently incompatible)
  - Configurable via `lambda_align` parameter (default: 0.0)

#### GAN Training Improvements
- **Fixed GAN gradient flow**: Added `.detach()` before decoder in adversarial loss
  - Prevents GAN gradients from corrupting encoder latent space
  - Encoder only learns from reconstruction+KL loss
  - Decoder learns from both reconstruction and GAN losses
- **Increased default GAN weight**: `lambda_adv: 0.05 → 0.1` for stronger discriminator signal

#### Training Robustness
- **Instance noise with exponential decay** for discriminator
  - Reduces mode collapse risk
  - Configurable via `instance_noise_std` and `instance_noise_decay`
- **Adaptive loss balancing** via inverse weighting
  - Automatic balancing of reconstruction, perceptual, and adversarial losses
  - Configurable via `adaptive_weights` parameter
- **Parameterized magic numbers**
  - `mse_weight` parameter for MSE loss weighting (default: 0.1)

#### Monitoring
- **Comprehensive metrics dashboard** with detailed training statistics
  - Reconstruction metrics (MSE, L1, frequency losses)
  - Perceptual metrics (LPIPS)
  - Adversarial metrics (generator, discriminator, R1 penalty)
  - Text alignment metrics
  - Adaptive loss weights

### Changed

#### Breaking Changes
- **FlowTrainer.train_step()** return type changed from `float` to `dict[str, float]`
  - **Before**: Returns single loss value
  - **After**: Returns comprehensive metrics dictionary
  - **Migration**: Update training scripts to handle dict return value
  - **Example**:
    ```python
    # Before
    loss = trainer.train_step(batch)

    # After
    metrics = trainer.train_step(batch)
    loss = metrics['flow_loss']  # Note: key is 'flow_loss', not 'loss'
    ```

#### Parameters
- **VAETrainer** new parameters (all have defaults, backward compatible):
  - `use_lpips=True` - Enable LPIPS perceptual loss
  - `lambda_lpips=0.1` - LPIPS loss weight
  - `instance_noise_std=0.01` - Initial instance noise std dev
  - `instance_noise_decay=0.9999` - Instance noise decay rate
  - `adaptive_weights=True` - Enable adaptive loss balancing
  - `mse_weight=0.1` - MSE reconstruction loss weight

- **FlowTrainer** new parameters (all have defaults, backward compatible):
  - `ema_decay=0.9999` - EMA decay rate for parameter averaging
  - `lambda_align=0.0` - Text-image alignment loss weight (disabled by default, see Removed section)

- **Batch timing** with `Xs/batch` in console output
- **Step-specific progress files** for pipeline mode
  - Each step writes to its own `training_metrics_<step_name>.jsonl`
- **Correct GAN loss keys** in logs
  - `loss_gen` (generator loss) and `loss_disc` (discriminator loss)
  - Previously logged with inconsistent keys
- **Mid-epoch sample generation** with batch numbers in filenames
  - Sample naming: `sample_<step>_epoch_<N>_batch_<M>.png`
  - Re-enabled after temporary disable in v0.1.x
- **Pipeline-aware diagram generation**
  - Generates separate diagrams per pipeline step
  - Aggregates metrics across steps for overview
- **Step-specific graphs**
  - Loss curves per step for focused analysis
  - Learning rate schedules per step
- **Max steps parameter** for quick testing
  - `max_steps` CLI arg and pipeline config
  - Exit training after N batches (useful for CI/testing)
- **Step/epoch/batch naming** for sample images
  - Clear provenance for generated samples
  - Easier correlation with training logs
- **YAML-first configuration** for pipeline mode
  - CLI args still supported for standard training
  - Pipeline mode requires YAML config file
- **Backward compatibility**
  - All existing CLI args still work
  - Standard training mode unchanged
- **New**: `docs/PIPELINE_ARCHITECTURE.md` (547 lines)
  - Complete pipeline training guide
  - Configuration reference with examples
  - Troubleshooting guide
  - GAN-only mode documentation
- **Updated**: `README.md`
  - Pipeline training mode section
  - GAN-only mode section
  - Enhanced console output examples
  - Sample naming conventions
  - v0.2.0 features highlighted
- **Updated**: `docs/TRAINING_GUIDE.md`
  - 100+ line pipeline training section
  - Quick start examples
  - Pipeline vs. standard training comparison
  - Complete 3-stage pipeline example
- **Updated**: `CONTRIBUTING.md`
  - Pipeline testing guidance
  - Step-by-step contribution workflow
- Added `lpips>=0.1.4` dependency for perceptual loss computation
- LPIPS requires VGG16 pretrained weights (~528MB download on first use)
  - Pre-download: `python -c "import lpips; lpips.LPIPS(net='vgg')"`
  - Cached in `~/.cache/torch/hub/checkpoints/`
- EMA parameters are not saved separately; use the tracked parameters for inference
- Adaptive weights are computed per-batch based on inverse loss magnitudes
- Instance noise decays to near-zero after ~10k steps

### Fixed

#### Bug Fixes
- **GAN-only mode fixes**
  - Fixed encoder gradients not flowing when `train_reconstruction=false`
  - Fixed VAE trainer not called when `train_vae=false` but GAN enabled
  - Fixed EMA not created for GAN-only mode
  - Fixed metrics/console logging for GAN-only (check buffer instead of `train_vae` flag)
- **Pipeline mode fixes**
  - Fixed checkpoint resume state tracking for multi-step pipelines
  - Fixed diagram generation for step-specific metrics files
  - Fixed FloatBuffer attribute error (`count` → `len(_items)`)
- **Sample generation fixes**
  - Fixed sample file renaming conflicts
  - Use epoch instead of batch in primary sample filenames
  - Add step/epoch/batch naming for clarity
- **LPIPS deprecation warning** - Suppressed torchvision `pretrained` parameter warnings during LPIPS initialization
- **Frequency-aware loss dimension mismatch** - Fixed `avg_pool2d` to use `kernel_size=3, padding=1` to preserve dimensions
- **Text-image alignment dimension mismatch** - Fixed tensor pooling and added dimension validation
- **FlowTrainer return type** - Training script now correctly handles dict return from `train_step()`
- **Text-image alignment disabled by default** - Changed `lambda_align` from `0.1` to `0.0` due to embedding dimension incompatibility
- **Batch size > 1 support** - Fixed normalization and cosine similarity dimension handling
- **Warning spam** - Alignment dimension mismatch warning only shows if feature is enabled

### Removed
- **Text-image alignment loss (disabled by default)** - Feature requires matching embedding dimensions between image (128D) and text (1024D) features
  - Set to `lambda_align=0.0` by default to avoid runtime errors
  - To enable: Add projection layer and set `lambda_align > 0`
  - Dimension mismatch is gracefully handled with warning
