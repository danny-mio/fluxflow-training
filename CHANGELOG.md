# Changelog

All notable changes to FluxFlow Training will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

Not yet released — work in progress toward v0.10.0.

### Added (Experimental)
- **AMD ROCm/gfx1151 support (experimental, unvalidated)**: `get_device()`
  now delegates to `fluxflow.utils.device` (distinguishes ROCm from real
  NVIDIA CUDA). `train.py` prints "Using ROCm backend (experimental)" when
  applicable, and the high-memory-usage warning is reworded on ROCm to
  clarify it reflects the VRAM/GTT carve-out, not total system RAM. New
  `--attention_backend {einsum,sdpa}` CLI flag / `model.attention_backend`
  config key (default `sdpa`, benchmarked as the fastest backend on ROCm,
  CUDA, and MPS; `einsum` remains available as a fallback; only affects
  v0.7.0/v0.8.0/v0.10.0 models built via the factory path). New `examples/config-rocm.yaml`
  starting-point config for Strix Halo. See `docs/ROCM.md` in fluxflow-core.
  Not yet empirically validated on real hardware.
- **Opt-in aspect-ratio bucketing** (`aspect_ratio_bucketing: bool`,
  per-dataset, default `false`) for `type: local` datasets: groups images by
  their exact post-transform target shape (`build_shape_dimension_cache`)
  instead of rounded native size, so every batch drawn from a group is
  shape-uniform batch-to-batch — existing behavior is unchanged when left
  off. Rejected by config validation when combined with `type: webdataset`
  or with per-dataset `reduced_min_sizes`. Mitigates pathological training
  slowdowns on ROCm targets with no precompiled MIOpen kernel database for
  the architecture (e.g. gfx1151), where every distinct convolution shape
  triggers a full kernel re-benchmark/recompile; not needed on backends
  without shape-keyed kernel autotuning (e.g. MPS, or CUDA with a populated
  kernel cache). Also: per-dataset `reduced_min_sizes` now takes precedence
  over the global `--reduced_min_sizes` value when set, matching the
  existing `batch_size`/`workers` per-dataset override convention.

### Added
- **`--precision {fp16,bf16}` / `training.precision` config key** (default
  `fp16`, only consulted when `use_fp16`/`--use_fp16` is true): opt-in bf16
  mixed precision, an immediate mitigation for the fp16 NaN/Inf crash the VAE
  expander hits mid-run as activations grow (fp16's ~65504 max clips to Inf;
  bf16 shares fp32's exponent range, so no overflow, at the cost of mantissa
  precision — an acceptable tradeoff for a range problem, not a precision
  one; the root-cause activation-growth fix lands separately in
  `fluxflow-core`). New `_resolve_mixed_precision(use_fp16, precision)`
  helper in `scripts/train.py` is the single choke point all four
  `Accelerator(mixed_precision=...)` construction call sites (both the
  legacy and pipeline training paths, CUDA/ROCm and CPU branches) now go
  through, replacing the hardcoded `"fp16" if args.use_fp16 else "no"`
  ternary at each site. Threaded through the same CLI-overrides-config
  fallback mechanism as `--use_fp16`. Fully backward compatible: any
  existing config with only `use_fp16: true` (no `precision` key) keeps
  resolving to fp16, unchanged. Set `training: {use_fp16: true, precision:
  bf16}` in YAML, or `--use_fp16 --precision bf16` on the CLI, to opt in.
- **Per-token text conditioning helpers**: `apply_cfg_null_substitution` in
  `training/cfg_utils.py` replaces zero-vector dropout with an encoded
  empty-prompt pair `(null_seq, null_mask)`. The pair is built once via
  `fluxflow.utils.visualization.build_cfg_null_pair` and cached on the text
  encoder, keeping train- and inference-time CFG distributions aligned.
- **Loss helpers** in `training/losses.py`:
  - `compute_ctx_shrinkage(ctx_features, alpha)` — β-VAE-style L2 pressure on
    deterministic ctx features at the bottleneck (v0.10.0 redesign §5.5).
  - `cosine_warmup_weight(step, warmup_steps, max_weight)` — KL_z warmup.
  - `delayed_cosine_warmup_weight(step, start_step, warmup_steps, max_weight)`
    — ctx-shrinkage delayed cosine warmup.
- **VAE trainer wiring**: a forward hook on `ctx_zinject_norm` captures the
  pre-attention ctx tensor each step; the trainer applies a delayed cosine
  warmup of `ctx_shrinkage_weight` and adds the term to the total VAE loss.
  KL_z uses `cosine_warmup_weight` over `kl_z_warmup_steps` toward
  `kl_z_weight`.
- **Flow trainer wiring**: consumes `(text_seq, text_mask)` per-token from
  `BertTextEncoder`. CFG dropout now routes through
  `apply_cfg_null_substitution` instead of zero-out. Polymorphic dispatch
  keeps v060/v070 legacy pooled-vector flows working.
- **New `PipelineStepConfig` keys** (see `config.example.yaml`):
  - `kl_z_weight` / `kl_z_warmup_steps` (clean Gaussian latent KL).
  - `ctx_shrinkage_weight` / `ctx_shrinkage_warmup_start_step` /
    `ctx_shrinkage_warmup_steps`.
  - `t_txt` (default `32` — DistilBERT max length per design §5.7).
  - `null_prompt` (empty-prompt string cached as the CFG null context).
  - `freeze_context_branch` — freeze `ctx_encoder_first_step`, `ctx_encoder_z`,
    `ctx_proj`, `ctx_token_attn`, `ctx_final_norm` while leaving the z-path
    trainable.
- **Freeze-list components**: `text_encoder_projection` and
  `text_encoder_backbone` allow targeting the DistilBERT `output_layer` and
  `language_model` sub-components independently. `context_branch` is also a
  valid freeze key for the new ctx encoder. The optimization validator
  rejects co-existence of the whole `text_encoder` optimizer key with the
  split sub-component keys.
- **Configurable dataset `max_text_length`**: dataset-level setting (default
  `32`) plumbed through `data/datasets.py` so tokenization matches the
  v0.10.0 `t_txt` budget. Older configs continue to load and emit a
  deprecation warning when they still pass `kl_beta` / `kl_warmup_steps`.
- **Acceptance test stubs** in `tests/acceptance/test_redesign_done_criteria.py`
  for the M8 retraining run. Marked `acceptance, gpu, slow`; skipped until a
  trained checkpoint and held-out prompts are wired in.
- **Text encoder is now bundled into the main checkpoint file**
  (`flxflow_final.safetensors`, `text_encoder.*`-prefixed keys) in addition
  to the existing standalone `text_encoder.safetensors` sibling file, which
  is still written unchanged. The sibling file takes precedence at load
  time — see `BertTextEncoder.load_with_override` in `fluxflow-core`.
- **`update_ratio_flow` diagnostic**: Lion update-norm/grad-norm ratio,
  logged alongside the console "Flow" metric so a fixed-amplitude Lion
  oscillation can be told apart from genuine convergence.
- **`optimizer_type` param on `get_default_optimizer_config`**: gives
  vae/text_encoder/discriminator Lion-appropriate lr/weight_decay defaults
  if a user opts one of them into Lion; their existing AdamW defaults are
  unchanged unless opted in.
- **`discriminator_update_freq` config knob** (default `1`, matching current
  every-step behavior) for the VAE/GAN pipeline step.
- **SPADE γ/β drift-from-init logging** (`spade_gamma_drift`,
  `spade_beta_drift`) and **`ctx_shrinkage_loss`/`ctx_aux_loss` logging**
  alongside the renamed ctx probe metric (see Fixed) — these two do carry
  gradient to SPADE, unlike the probe.
- **Prompt-truncation diagnostics**: sampled (1-in-50) truncation-rate and
  caption-length-percentile logging across all three dataset classes.
  `max_text_length` is now a real `--max_text_length` CLI flag /
  `data.max_text_length` YAML key (previously hardcoded to `32` with no
  override); default unchanged.

### Changed
- **Flow trainer text contract**: pooled `[B, D]` text vectors are replaced
  by `(text_seq [B, T_txt, D], text_mask [B, T_txt])`. The trainer detects
  the active model version and dispatches to the legacy code path for v060
  / v070 checkpoints.
- **KL_z schedule**: KL warmup is now a cosine ramp over `kl_z_warmup_steps`
  toward `kl_z_weight` (legacy `kl_beta` / `kl_warmup_steps` still load with
  a deprecation warning; v0.10.0 keys win when both are present).
- **`ctx_loss_weight` default**: changed from `1.0` (v0.8.1) to `0.5` per
  the v0.10.0 DP-1 decision balancing ctx-dim and VAE-dim v-prediction
  losses in `FlowTrainer`.
- **`lambda_ctx_aux` default**: `0.01` (auxiliary stop-grad MSE on ctx vs
  `z_tokens` during VAE training, per plan §4.1 DP-1).
- **`fluxflow` dependency bumped to `>=0.10.0`**: required for
  `fluxflow.utils.visualization.build_cfg_null_pair` and the v0.10.0 model
  architecture (per-token text, ctx_zinject_norm, multi-scale SPADE).
- **Flow Lion hyperparameters retuned**: `lr` `5e-7` → `1e-7`,
  `weight_decay` `0.01` → `0.05`; cosine schedule floor (`eta_min_factor`)
  `0.1` → `0.01` for flow. Root cause: Lion previously shared AdamW's
  learning rate, a known Lion misconfiguration (Lion needs roughly 5-10x
  smaller lr and larger weight_decay than AdamW), producing a fixed-point
  plateau instead of convergence.
- **`kl_z_weight`/`ctx_shrinkage_weight` now actually wired into
  `VAETrainer`** (`pipeline_orchestrator.py`, `scripts/train.py`) — these
  `PipelineStepConfig` fields existed but were previously never passed
  through, so setting them had zero effect. **Behavior change**: any
  pipeline-mode step with `train_vae: true` that doesn't explicitly
  override these two keys now switches from the prior real behavior
  (legacy `kl_beta` KL path, inactive ctx-shrinkage) to the ratified
  v0.10.0 defaults actually taking effect (`kl_z_weight=0.5` cosine-warmup
  KL_z, `ctx_shrinkage_weight=0.001` active ctx-shrinkage). The legacy
  `scripts/train.py` CLI driver's new flags default to `0.0` instead, so
  that path has no behavior change unless explicitly opted in.

### Deprecated
- `kl_beta` — use `kl_z_weight`.
- `kl_warmup_steps` — use `kl_z_warmup_steps`.

Both legacy keys still load via `_parse_step_config` and emit a
`DeprecationWarning`. The new key wins when both are present.

### Fixed
- **Mid-training resume can silently zero FiLM conditioning on a pre-Fix-B
  checkpoint**: fluxflow-core's `FluxTransformerBlock_v100` gained zero-init
  `film_text_scale`/`film_time_scale` params so dual FiLM starts as an exact
  identity instead of a random Xavier perturbation (Plan02 Fix B). A
  checkpoint saved before that fix has no such keys; `train_legacy`'s resume
  path loads `flow_processor` with `strict=False`, so the missing keys
  silently default to 0 — completely zeroing (not merely attenuating) all
  timestep/text FiLM conditioning model-wide, with no error or log to flag
  it. `train_legacy` now warns immediately after the `flow_processor`
  checkpoint load whenever `film_text_scale`/`film_time_scale` are absent
  from the loaded state dict. If you resume a pre-Fix-B v0.10.0-line flow
  checkpoint, expect this warning once and plan for a brief fine-tuning
  pass to recover the conditioning.
- `CheckpointManager.save_models`'s embedded safetensors metadata
  (`model_version`/`model_type`/`vae_dim`) was silently never written when
  `model_config` was a pydantic `ModelConfig` object (the real-data call
  site) rather than a plain dict — `"model_version" in model_config` on a
  `BaseModel` evaluates `False` instead of raising, since `__iter__` yields
  `(name, value)` tuples, not field names. `model_config` is now normalized
  via `model_dump()` when available.
- **Aspect-ratio bucketing's shape locality**: `ResumableDimensionSampler`
  shuffled batches across *all* size groups each epoch, so consecutive
  batches almost never shared a shape even with `aspect_ratio_bucketing`
  enabled — defeating the point (MIOpen kernel-cache reuse). New opt-in
  `group_contiguous` sampler param keeps all batches of one group
  contiguous, shuffling only group *visitation* order (and image order
  within each group); wired on automatically for `aspect_ratio_bucketing`
  datasets, unchanged (off) otherwise.
- **Training loop crash on ROCm GAN discriminator step**:
  `_train_discriminator`'s R1 gradient penalty
  (`torch.autograd.grad(..., create_graph=True)`, a double-backward)
  reliably triggers a MIOpen GEMM/Im2Col solver that fails to compile under
  HIPRTC on gfx1151, crashing the whole run every `r1_interval` steps.
  `_train_discriminator` now catches this specific `RuntimeError` (matched
  narrowly on `"miopen"` in the message — other errors still propagate),
  skips the discriminator's optimizer step for that iteration, and reports
  it via the existing `_optimizer_stepped` skip convention already used for
  NaN-guarded generator steps, so it doesn't pollute loss buffers or step
  the discriminator scheduler. Separately, the crash-logging call in the
  training loop used `logger.error(...)`, which never reaches this
  project's `train.log` (only `print()` output does) — changed to
  `print()` so a crash's step/batch/shape context is actually visible.
- **Mislabeled `context_alignment` metric renamed to `ctx_probe_alignment`**
  and clarified in comments as a fully-detached diagnostic probe with no
  gradient path to SPADE — not to be confused with the console "Ctx"
  metric, which is `flow_trainer.py`'s own context-dim loss.
- **Silently-swallowed `except Exception` around `ctx_aux_loss`**, which
  used to zero the loss with only a generic warning, now logs the real
  exception.
- **LR scheduler paced against a massively-inflated step budget, pinning LR
  near its initial max for the practical duration of a run**:
  `batches_per_epoch` was computed from the top-level `training.batch_size`
  instead of the per-dataset `batch_size` override (e.g. a step's dataset
  sets `batch_size: 8` against a top-level `batch_size: 1`), and the
  `total_steps` passed to `_create_step_schedulers` was never divided by
  `gradient_accumulation_steps`. Combined, this could inflate a scheduler's
  step budget (e.g. `CosineAnnealingLR`'s `T_max`) by roughly
  `(dataset_batch_size / top_level_batch_size) × gradient_accumulation_steps`
  — ~160x in the reported real-world case. Previously misdiagnosed as a
  Lion-optimizer-specific "fixed point" issue; it isn't optimizer-specific,
  since the scheduler code path is shared by AdamW and Lion alike. New
  shared helpers `resolve_dataset_batch_size()` and
  `compute_batches_per_epoch()` (ceil-based, replacing floor-division)
  replace the duplicated/wrong inline math at all three call sites. The
  `gradient_accumulation_steps` division is deliberately scoped to only the
  flow/text-encoder schedulers inside `_create_step_schedulers` — `vae` and
  `discriminator` are excluded, because `VAETrainer.scheduler.step()` fires
  every micro-batch with no accumulation gating at all (`VAETrainer` doesn't
  even accept a `gradient_accumulation_steps` param); dividing globally
  would have introduced a new premature-LR-decay bug for any step with
  `train_vae=True`. The same wrong `batches_per_epoch` also fed
  `epoch_total_batches` (gates the batch loop's early-break and resume
  fast-forward logic) — in the opposite imbalance case (top-level
  `batch_size` larger than a dataset's per-dataset `batch_size`), this
  would have truncated real epochs early.
- **`VAETrainer` fp16/GradScaler crash and silent gradient corruption**:
  `training.use_fp16: true` had never been exercised by any prior run (every
  config touching the VAE trainer kept it `false`), so two GradScaler bugs
  went unnoticed until a real fp16 run hit them.
  - `_train_generator` crashed on the first batch with `RuntimeError:
    unscale_() has already been called on this optimizer since the last
    update()`. `self.optimizer` is a plain `torch.optim` instance built
    directly in `pipeline_orchestrator.py::_create_step_optimizers` (never
    passed through `accelerator.prepare()`), and the method called
    `self.accelerator.scaler.unscale_(self.optimizer)` manually and then
    `self.accelerator.clip_grad_norm_(...)`, which unscales the same
    optimizer again internally — with no `scaler.update()` anywhere to
    close the unscale→update cycle. Fixed by unscaling once via
    `accelerator.unscale_gradients(self.optimizer)`, clipping with plain
    `torch.nn.utils.clip_grad_norm_()` instead of the accelerator wrapper
    (which would re-unscale), and calling `scaler.update()` after the
    optimizer step.
  - `_train_discriminator` had the same missing-unscale pattern but with no
    crash to flag it: `discriminator_optimizer.step()` ran with no unscale
    at all under fp16, so gradients were silently applied still multiplied
    by the GradScaler's growth factor (≥2^16) — the first `gan_training:
    true` run under fp16 would have diverged almost instantly, with nothing
    pointing at a bug rather than bad hyperparameters. Fixed with the same
    unscale→step→update sequence as the generator fix.
- **`VAETrainer` never actually accumulated gradients despite
  `gradient_accumulation_steps` being configured on every VAE pipeline
  step**: `_train_generator`/`_train_discriminator` called
  `zero_grad()`/`optimizer.step()`/scheduler `.step()`/`ema.update()`
  unconditionally on every micro-batch instead of once per real
  accumulation boundary, so the actual effective batch size — the thing
  the GAN-stability hyperparameters in the from-scratch training configs
  were tuned around — was silently far smaller than configured. This is
  what `pipeline_orchestrator.py`'s now-removed comment excluding
  `"vae"`/`"discriminator"` from scheduler-budget division was
  documenting as known behavior: a previous fix in this same effort built
  a compensating workaround around this bug rather than catching it (see
  the scheduler-pacing fix above). Fixed by mirroring `FlowTrainer`'s
  `_accumulation_step`/`should_step` pattern: generator and discriminator
  now share one boundary counter so both step in lockstep — since
  `VAETrainer`'s dual-optimizer structure doesn't exist in `FlowTrainer`,
  the discriminator reads the shared counter *before* the generator
  increments it, computing the same boundary value the generator computes
  *after* incrementing. Loss is now divided by
  `gradient_accumulation_steps` before `backward()`, matching
  `FlowTrainer`. `pipeline_orchestrator.py`'s
  `_ACCUMULATION_GATED_SCHEDULER_NAMES` now includes `"vae"` and
  `"discriminator"` so their scheduler budgets are divided like
  `FlowTrainer`'s; the two scheduler-pacing regression tests that
  asserted the old undivided behavior are renamed and updated to assert
  the correct divided behavior (they were locking in the bug, not
  weakening coverage). Known follow-up: `discriminator_update_freq > 1`
  combined with `gradient_accumulation_steps > 1` can theoretically leave
  discriminator gradients silently discarded mid-window; doesn't affect
  any current config (all use the default `discriminator_update_freq=1`).
- **`use_fp16`/`mixed_precision="fp16"` never actually engaged
  mixed-precision compute anywhere**, on either trainer: no
  `torch.autocast`/`accelerator.autocast()` context existed in
  `vae_trainer.py` or `flow_trainer.py`, and `Accelerator(mixed_precision=
  "fp16")` was constructed but models were never passed through
  `accelerator.prepare()` (only dataloaders were) — so nothing ever
  actually ran in fp16. Enabling the flag only activated `GradScaler`'s
  unscale/inf-check bookkeeping overhead for a precision reduction that
  never happened, fully explaining a user report that `use_fp16` measured
  *slower* than fp32 on a real A6000/Ampere GPU: pure overhead, zero
  benefit, on any hardware. Fixed by adding an `_autocast()` helper
  (`nullcontext()` fallback without an accelerator) to both trainers and
  wrapping the compressor/expander/discriminator forward passes in
  `vae_trainer.py` and the compressor/flow_processor forward passes in
  `flow_trainer.py`. LPIPS and the small `context_predictor` diagnostic
  head are deliberately kept OUTSIDE autocast with an explicit `.float()`
  cast on their inputs — a tensor produced inside an autocast context
  keeps its realized fp16 dtype after leaving the context, and LPIPS's
  VGG backbone has fp32 weights, so this would otherwise be a real crash
  (confirmed via a real-GPU test during implementation, not theoretical).
  `text_encoder`'s forward pass is not yet wrapped in autocast; flagged as
  a natural follow-up, out of scope here. 20 new tests across
  `test_vae_trainer_gradient_accumulation.py` (12: step/zero_grad call
  counts, loss scaling, gradient survival mid-window, EMA/scheduler
  gating, generator/discriminator lockstep), `test_vae_trainer_autocast.py`
  (5), and `test_flow_trainer_autocast.py` (3, including real-GPU tests on
  ROCm/HIP asserting actual `torch.float16` activation dtype). Full
  `tests/unit/` suite passes (702 pre-existing + 20 new, 2 gpu/slow
  deselected); black/isort/flake8 clean; mypy's 63 pre-existing errors are
  unchanged (verified via git-stash A/B diff), zero new errors introduced.

### Migration
- Bump configs to the v0.10.0 schema: rename `kl_beta` → `kl_z_weight`,
  `kl_warmup_steps` → `kl_z_warmup_steps`, and set `t_txt: 32` on flow
  steps. See the v0.10.0 redesign migration notes in fluxflow-core
  (<https://github.com/danny-mio/fluxflow-core/blob/develop/docs/MIGRATION-v0.10.0-redesign.md>)
  for the full cross-package walkthrough.

## [0.8.1] - 2026-04-03

### Added
- **`examples/config-mps.yaml`**: Recommended training config for Apple Silicon (`batch_size=2`, `use_fp16=false`, `use_gradient_checkpointing=true`, `workers=0`).
- **`scripts/profile_mps.py`**: Profiles CPU fallbacks during a VAE forward+backward pass on MPS. Run on Apple Silicon to identify remaining bottlenecks.
- **`ctx_loss` console logging**: Context dim loss is now shown in the training progress line (` | Ctx: 0.0123`) alongside the flow loss.
- **`ctx_loss` graph curve**: Context dim loss is plotted as a separate curve in `training_losses.png` and the combined overview.
- **`ctx_loss_weight` config param**: Allows tuning the relative weight of the context-dim v-prediction loss in the flow training config.

### Fixed
- **MPS cache clearing**: `torch.mps.empty_cache()` added alongside `torch.cuda.empty_cache()` in VAETrainer and FlowTrainer.
- **MPS adaptive pooling**: Replaced ad-hoc try/except in VAETrainer with `mps_safe_pool2d` from `fluxflow.utils.mps`.
- **v-prediction loss target**: Flow trainer was computing loss against clean x0 despite `prediction_type="v_prediction"`. Now correctly computes `v = alpha_t * noise - sigma_t * x0` using `alphas_cumprod[t]`.
- **Context dim train/inference mismatch**: Previously only VAE dims were noised during training while context dims were passed clean, but inference denoises all 133 dims from noise. Training now noises all dims uniformly to match inference. The trivial x0-reconstruction ctx_loss is replaced with v-prediction over all dims.
- **Context dim loss scale**: VAE dims (~unit Gaussian) and context dims (~0.1–0.5 range) are normalised independently to prevent a ~10x loss imbalance. A `ctx_loss_weight` knob (default `1.0`) allows tuning.
- **Gradient clipping**: Replaced a broken adaptive clipping formula (`min(clip_norm, norm*1.5)`) with a straightforward `clip_grad_norm_` call. Removes the redundant manual norm loop.
- **GAN instance noise never applied**: `add_instance_noise` had an inverted guard (`if not x.requires_grad: return x`) that silently skipped noise on all discriminator inputs (raw batch and detached tensors always have `requires_grad=False`). Guard removed; noise now always applied.
- **GAN adaptive weight explosion at startup**: `_compute_adaptive_weight` computed `target / avg` with near-zero `avg` for the GAN loss at the start of adversarial training, yielding 300–3000× amplification and explosive gradients. Now clamped to `max_weight=5.0`.
- **Discriminator trained on deterministic latents**: `_train_discriminator` called `compressor(real_imgs, training=False)` which returns a deterministic tensor; the generator always sees stochastic samples. Changed to `training=True` (wrapped in `torch.no_grad()`) so real images are encoded with reparameterisation, matching the distribution the generator learns against.
- **ctx_loss normalisation scale**: Flow trainer normalised per-group losses by the clean-x0 std of each group (~0.1–0.5 for context dims). At noisy timesteps `v_target ≈ noise` (std ~1.0), this amplified `ctx_loss` ~11× relative to `vae_loss`. Now normalises by the v-target's own std so scale is consistent across all timesteps.

### Changed
- **`fluxflow` dependency bumped to `>=0.8.1`**: Required for `fluxflow.utils.mps.mps_safe_pool2d`.

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
