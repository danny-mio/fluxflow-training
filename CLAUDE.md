# CLAUDE.md - FluxFlow Training

See `AGENTS.md` for the canonical agent and coordinator conventions used in
this repo. This file holds Claude-specific notes layered on top of that.

## v0.10.0 Bezier-Coupled Redesign (in progress on feature/model-v0.10.0)

The v0.10.0 redesign coordinates with `fluxflow-core` v0.10.0 and ships the
training-side glue for the five locked decisions:

1. **Per-token text** — flow trainer consumes `(text_seq, text_mask)` from
   `BertTextEncoder` instead of pooled vectors.
2. **Conditional ctx coupling** — the VAE trains ctx with `ctx_dropout_frac` /
   `ctx_noise_sigma` (the `ctx_shrinkage_weight` term was removed as inert).
3. **Full flow modernization** — flow trainer wires 2D RoPE + dual FiLM
   downstream of `FluxFlowProcessor_v100`.
4. **Multi-scale SPADE** — VAE trainer respects the new decoder topology with
   no config changes required beyond the latent-dim drop.
5. **Clean Gaussian z** — `kl_z_weight` (cosine warmup) and `data.max_text_length` (text-token length, default 128 via
   `fluxflow.text_length.DEFAULT_MAX_TEXT_LENGTH`) are exposed in the YAML config.

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

Plans live in `fluxflow-core/docs/plans/2026-06-13-v0.10.0-redesign-design.md`
and `-implementation.md`. Per-repo milestone tag: `m5-training-pipeline`.

Acceptance criteria stubs: `tests/acceptance/test_redesign_done_criteria.py`
(marked `acceptance, gpu, slow`; filled in after the M8 retraining run).
