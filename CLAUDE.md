# CLAUDE.md - FluxFlow Training

See `AGENTS.md` for the canonical agent and coordinator conventions used in
this repo. This file holds Claude-specific notes layered on top of that.

## v0.10.0 Bezier-Coupled Redesign (in progress on feature/model-v0.10.0)

The v0.10.0 redesign coordinates with `fluxflow-core` v0.10.0. Five locked
decisions: fluxflow-core `docs/ARCHITECTURE.md` § "v0.10.0: Bezier-Coupled
Architecture". Training workflow: `docs/TRAINING_GUIDE.md` § "Training workflow
(v0.10.0)". Text length: `data.max_text_length` (default 128 via
`fluxflow.text_length.DEFAULT_MAX_TEXT_LENGTH`). `ctx_shrinkage_*` removed (inert).

New config keys (see `config.example.yaml`):
- `kl_z_weight` — final KL weight on the clean Gaussian latent (default 1e-5; per-element KL). `kl_z_warmup_steps` counts optimizer steps.
- `kl_free_bits` — per-channel KL floor, nats/element (default 0.5).
- `calibration_images` — latent-stat calibration at the VAE→flow transition (default 2000). Workflow: `docs/TRAINING_GUIDE.md` § "Training workflow (v0.10.0)".
- `hf_ramp_steps` — HF-loss alpha ramp length (default 30000 micro-batch steps).
- Logs: `VAE`/`vae_loss` is HF-weighted recon over the whole batch incl. the z-only half (rises as alpha ramps); watch `L1ctx` (plain ctx-path L1).
- `t_txt` — deprecated, ignored (warns). Text length = `data.max_text_length`.

CFG empty-prompt substitution at flow-train time goes through
`fluxflow_training.training.cfg_utils.apply_cfg_null_substitution`
(pairs with `fluxflow.utils.visualization.build_cfg_null_pair`).

Per-repo milestone tag: `m5-training-pipeline`.

Acceptance criteria stubs: `tests/acceptance/test_redesign_done_criteria.py`
(marked `acceptance, gpu, slow`; filled in after the M8 retraining run).
